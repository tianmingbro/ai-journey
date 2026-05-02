# rag_evaluator.py
import os
import sys
import time
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from study.chunking_strategies import STRATEGIES
import re

def extract_doc_id_from_source(source: str) -> str:
    """从文件路径中提取 doc_id，如 'test_docs/doc_001_llm_intro.txt' -> 'doc_001'"""
    basename = os.path.basename(source)
    match = re.search(r'doc_\d+', basename)
    return match.group(0) if match else "unknown"
# ---------- 加载原始文档 ----------
DOC_DIR = "./test_docs"
# 确保可以导入 Day17 的文档处理器
# sys.path.append("../day17")
from study.document_processor import process_documents

raw_docs = process_documents(DOC_DIR, chunk_size=9999)  # 不做分块，返回原始文档列表
# 根据文件名自动设置 doc_id
for i, doc in enumerate(raw_docs, 1):
    doc.metadata["doc_id"] = f"doc_{i:03d}"
for doc in raw_docs:
    print(f"  {doc.metadata.get('filename', '?')} → {doc.metadata['doc_id']}")
print(f"原始文档数: {len(raw_docs)}")

# ---------- 评测问题集 ----------
TEST_QUESTIONS = [
    {"question": "大语言模型通常包含多少参数？", "relevant_doc_id": "doc_001"},
    {"question": "Transformer架构的核心组件是什么？", "relevant_doc_id": "doc_004"},
    {"question": "什么是向量数据库？", "relevant_doc_id": "doc_003"},
    {"question": "Python在AI领域的地位如何？", "relevant_doc_id": "doc_005"},
]

# ---------- 评测指标计算函数 ----------
# def evaluate_hit_rate(retrieved_docs: List[Document], relevant_id: str) -> float:
#     """命中率：相关文档是否出现在 Top-K 检索结果中 (K=3)"""
#     for doc in retrieved_docs:
#         if doc.metadata.get("doc_id") == relevant_id:
#             return 1.0
#         return 0.0
def evaluate_hit_rate(retrieved_docs: List[Document], relevant_id: str) -> float:
    for doc in retrieved_docs:
        if doc.metadata.get("doc_id") == relevant_id:
            return 1.0
    return 0.0
def evaluate_mrr(retrieved_docs: List[Document], relevant_id: str) -> float:
    """MRR：第一个相关文档排名的倒数"""
    for i, doc in enumerate(retrieved_docs, 1):
        if doc.metadata.get("doc_id") == relevant_id:
            return 1.0 / i
    return 0.0

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    "你是一个评估助手。请比较以下「根据上下文生成的回答」是否严格忠实于「上下文」。\n"
    "如果完全忠实，输出1.0；如果大部分忠实但有轻微偏差，输出0.7；"
    "如果部分忠实但有较明显错误，输出0.3；如果完全不符，输出0.0。\n"
    "只输出一个数字，不要输出其他内容。\n\n"
    "上下文：\n{context}\n\n回答：\n{answer}\n"
    "评分："
)
faithfulness_llm = ChatOpenAI(
    temperature=0, 
    model=os.getenv("MODEL_NAME","qwen-turbo"),           
    api_key=os.getenv("DASHSCOPE_API_KEY"),              
    base_url=os.getenv("DASHSCOPE_BASE_URL"),            
)

def evaluate_faithfulness(context: str, answer: str) -> float:
    """使用 LLM-as-Judge 评估生成答案的忠实度"""
    try:
        score_str = (FAITHFULNESS_PROMPT | faithfulness_llm | StrOutputParser()).invoke({
            "context": context, "answer": answer
        })
        return float(score_str.strip())
    except Exception:
        return 0.0

# ---------- 实验执行函数 ----------
def run_experiment(strategy_name: str, splitter_fn) -> Dict:
    """对指定分块策略执行完整评测，返回汇总指标与详细结果"""
    # 1. 分块
    chunks = splitter_fn(raw_docs)
    print(f"\n{'='*60}\n策略: {strategy_name}\n分块数: {len(chunks)}")
    # 根据文件名重新为每个 chunk 设置正确的 doc_id，确保万无一失
    for chunk in chunks:
        chunk.metadata['doc_id'] = extract_doc_id_from_source(chunk.metadata.get('source', ''))
    # 2. 构建向量库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model="text-embedding-v3",
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            check_embedding_ctx_length=False,
            chunk_size=10),
        persist_directory=f"./chroma_db_{strategy_name.replace(' ','_')}",
        collection_name="eval_kb",
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 构建 RAG 生成链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "基于上下文回答。\n上下文：\n{context}"),
        ("human", "{question}")
    ])
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": itemgetter("question")
        }
        | prompt
        | ChatOpenAI(
            temperature=0, 
            model=os.getenv("MODEL_NAME","qwen-turbo"),           
            api_key=os.getenv("DASHSCOPE_API_KEY"),              
            base_url=os.getenv("DASHSCOPE_BASE_URL"),            
        )
        | StrOutputParser()
    )

    # 4. 逐问题评测
    results = []
    for q in TEST_QUESTIONS:
        t0 = time.time()
        retrieved = retriever.invoke(q["question"])
        answer = rag_chain.invoke({"question": q["question"]})
        elapsed = time.time() - t0
        print(f"\n--- 调试: 问题 '{q['question'][:30]}...' 的检索结果 ---")
        for i, doc in enumerate(retrieved):
            print(f"检索结果 #{i+1}: id={doc.id}, metadata={doc.metadata}")
        print(f"期望的 relevant_doc_id: {q['relevant_doc_id']}")
        hit = evaluate_hit_rate(retrieved, q["relevant_doc_id"])
        # 在计算 hit 之后添加
        print(f"  手动验证: {any(doc.metadata.get('doc_id') == q['relevant_doc_id'] for doc in retrieved)}")
        mrr = evaluate_mrr(retrieved, q["relevant_doc_id"])
        faith = evaluate_faithfulness(
            "\n\n".join(d.page_content for d in retrieved), answer
        )

        results.append({
            "question": q["question"],
            "hit": hit,
            "mrr": mrr,
            "faithfulness": faith,
            "latency": elapsed,
            "answer": answer
        })
        print(f"  Q: {q['question'][:30]}... Hit={hit} MRR={mrr:.3f} Faith={faith:.2f}")

    # 计算平均值
    avg_hit = sum(r["hit"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"  平均: Hit Rate={avg_hit:.2%} MRR={avg_mrr:.3f} Faith={avg_faith:.2f} Latency={avg_latency:.2f}s")
    return {
        "strategy": strategy_name,
        "chunk_count": len(chunks),
        "avg_hit": avg_hit,
        "avg_mrr": avg_mrr,
        "avg_faith": avg_faith,
        "avg_latency": avg_latency,
        "details": results
    }

# ---------- 主程序 ----------
if __name__ == "__main__":
    all_results = []
    for name, fn in STRATEGIES.items():
        all_results.append(run_experiment(name, fn))

    # 输出最终对比表格
    print(f"\n{'='*70}\n📊 最终对比表格\n{'='*70}")
    print(f"{'策略':<25} {'Hit Rate':>10} {'MRR':>10} {'Faith':>8} {'Latency':>10} {'Chunks':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['avg_hit']:>9.2%} {r['avg_mrr']:>10.3f} {r['avg_faith']:>8.2f} {r['avg_latency']:>9.2f}s {r['chunk_count']:>8}")

    # 保存原始数据到 JSON 文件，供报告使用
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n✅ 评测完成，结果已保存至 evaluation_results.json")