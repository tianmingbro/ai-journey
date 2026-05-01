import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

# --------- LangSmith (可选) ---------
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-full-pipeline")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from typing import List, Dict

# 引入之前封装好的处理函数
# sys.path.append("./week4/day17")
from study.document_processor import process_documents

# --------- 1. 文档加载与预处理 ---------
DOC_DIR = "./test_docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

print("=" * 60)
print("🔄 文档加载与分块 (process_documents)")
t0 = time.time()
chunks = process_documents(DOC_DIR, chunk_size=400, chunk_overlap=80)
print(f"   耗时: {time.time()-t0:.2f} 秒, 共 {len(chunks)} 个文本块\n")

# --------- 2. 嵌入与持久化 ---------
print("📊 向量化与入库 (Chroma)")
embeddings = OpenAIEmbeddings(model="text-embedding-v3",api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            check_embedding_ctx_length=False)
t1 = time.time()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
)
print(f"   耗时: {time.time()-t1:.2f} 秒")

# --------- 3. 构建 RAG 链 ---------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "qwen-turbo"),           # 你的对话模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),               # 统一使用 api_key 参数
    base_url=os.getenv("DASHSCOPE_BASE_URL"),             # 统一使用 base_url 参数
)
def format_docs(docs: List[Document]) -> str:
    """将检索文档拼接为上下文字符串，附带来源标注"""
    return "\n\n".join(
        f"[来源 {i+1}: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识库问答助手。严格根据上下文回答，若无法回答请说明。\n上下文：\n{context}"),
    ("human", "{question}")
])


rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)
# --------- 4. 测试 ---------
print("=" * 60)
print("🧪 开始问答测试")
queries = [
    "大语言模型通常有多少参数？",
    "什么是向量数据库？",
    "今天天气怎么样？"
]
for q in queries:
    t_start = time.time()
    answer = rag_chain.invoke(q)
    print(f"\n❓ {q}")
    print(f"🤖 {answer}")
    print(f"⏱️ 耗时: {time.time()-t_start:.2f} 秒")

print("\n✅ 完整 RAG 流水线执行成功。")