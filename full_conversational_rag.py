#!/usr/bin/env python3
"""
完整记忆型 RAG 管线：文档加载 -> 分块 -> 入库 -> 检索 -> 生成 + 对话记忆
使用 ChatMessageHistory + LCEL 实现，零废弃类。
运行方式：python full_conversational_rag.py
"""

import os
import time
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

# LangChain 相关
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

# ------------------------------
# 1. 文档处理函数（Day17 缝合）
# ------------------------------
def process_documents(docs_dir: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[Document]:
    """加载目录中的 .txt/.md 文件，返回分割好的文档块"""
    print(f"📂 正在加载文档目录: {docs_dir}")
    loader = DirectoryLoader(
        path=docs_dir,
        glob="**/*.{txt,md}",
        loader_cls=lambda fp: TextLoader(fp, encoding='utf-8'),
        show_progress=True,
        use_multithreading=True
    )
    raw_docs = loader.load()
    print(f"   原始文档数: {len(raw_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    # 补充元数据
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["filename"] = Path(source).name
    print(f"   分割后文本块数: {len(chunks)}\n")
    return chunks


# ------------------------------
# 2. 初始化或加载向量数据库
# ------------------------------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
DOCS_DIR = "./test_docs"

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    check_embedding_ctx_length=False)

if os.path.exists(CHROMA_DIR) and os.path.isdir(CHROMA_DIR):
    print("✅ 检测到已有向量数据库，直接加载...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
else:
    print("🆕 未检测到数据库，开始创建...")
    if not os.path.isdir(DOCS_DIR):
        print(f"❌ 文档目录 {DOCS_DIR} 不存在！请先创建并放入 .txt/.md 文件。")
        exit(1)
    chunks = process_documents(DOCS_DIR)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"💾 向量数据库已创建，共 {len(chunks)} 个块。")

# 检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------------------
# 3. LLM 与格式化函数
# ------------------------------
llm = ChatOpenAI(
    temperature=0, 
    model=os.getenv("MODEL_NAME","qwen-turbo"),           
    api_key=os.getenv("DASHSCOPE_API_KEY"),              
    base_url=os.getenv("DASHSCOPE_BASE_URL"),            
)

def format_docs(docs: List[Document]) -> str:
    """将检索到的文档拼接为上下文字符串"""
    return "\n\n".join(
        f"[来源: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

# ------------------------------
# 4. 对话记忆管理（Day19 核心）
# ------------------------------
session_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def load_history(input_dict: dict) -> dict:
    """从存储中读取chat_history并注入到输入字典"""
    session_id = input_dict.get("session_id", "default")
    history = get_session_history(session_id)
    # 可选：限制窗口大小，避免 token 爆炸（保留最近 10 条消息）
    # if len(history.messages) > 10:
    #     history.messages = history.messages[-10:]
    return {**input_dict, "chat_history": history.messages}

# ------------------------------
# 5. 带记忆的 RAG 链
# ------------------------------
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system",
        "你是一个智能助手，能结合对话历史和检索知识回答问题。\n"
        "如果检索上下文不足以回答，请如实说明。\n\n"
        "检索上下文：\n{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

conversational_rag_chain = (
    RunnablePassthrough.assign(
        chat_history=RunnableLambda(load_history) | (lambda x: x["chat_history"]),
        context=RunnableLambda(lambda x: x["question"]) | retriever | format_docs    )
    | prompt_with_history
    | llm
    | StrOutputParser()
)

# ------------------------------
# 6. 问答交互与记忆保存
# ------------------------------
def ask_question(session_id: str, question: str) -> str:
    history = get_session_history(session_id)
    response = conversational_rag_chain.invoke({
        "session_id": session_id,
        "question": question
    })
    # 存入记忆
    history.add_user_message(question)
    history.add_ai_message(response)
    return response

# ------------------------------
# 7. 交互式主循环（含简单压测）
# ------------------------------
if __name__ == "__main__":
    print("\n🤖 带记忆的 RAG 助手已启动！")
    print("你可以像跟朋友聊天一样连续提问，系统会根据上下文和知识库回答。")
    print("输入 'clear' 清空记忆，输入 'exit' 退出。")

    session = "default_session"
    while True:
        user_input = input("\n👤 你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "clear":
            session_store[session] = ChatMessageHistory()
            print("🧹 对话记忆已清空。")
            continue

        # 计时
        start = time.time()
        answer = ask_question(session, user_input)
        elapsed = time.time() - start

        print(f"🤖 助手: {answer}")
        print(f"⏱️  耗时: {elapsed:.2f} 秒")

    # 简单压力测试：连续发送 3 个问题
    print("\n📊 正在执行简单压力测试（连续 3 个问题）...")
    test_questions = [
        "大语言模型是什么？",
        "它有什么特点？",
        "那它的训练需要什么资源？"
    ]
    for i, q in enumerate(test_questions, 1):
        t0 = time.time()
        ans = ask_question("stress_test", q)
        t1 = time.time()
        print(f"  Q{i}: {q}")
        print(f"  A{i}: {ans[:80]}...")
        print(f"  耗时: {t1-t0:.2f} 秒")
    print("✅ 压力测试完成。")