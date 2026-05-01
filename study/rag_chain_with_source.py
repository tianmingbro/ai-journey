import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from typing import List

# ------------------- ChromaManager -------------------
class ChromaManager:
    def __init__(self, persist_directory="./chroma_db", collection_name="knowledge_base"):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-v3",                     # 阿里云模型名称
            api_key=os.getenv("DASHSCOPE_API_KEY"),        # 确保环境变量正确
            base_url=os.getenv("DASHSCOPE_BASE_URL"), 
            check_embedding_ctx_length=False
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def as_retriever(self, k=3):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

manager = ChromaManager()
retriever = manager.as_retriever(k=3)

# ------------------- LLM -------------------
llm = ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "qwen-turbo"),           # 你的对话模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),               # 统一使用 api_key 参数
    base_url=os.getenv("DASHSCOPE_BASE_URL"),             # 统一使用 base_url 参数
)

# ------------------- 格式化函数 -------------------
def format_docs_with_source(docs: List[Document]) -> str:
    """将文档内容和文件来源拼接为上下文字符串"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知文件")
        parts.append(f"[来源{i} - {source}]\n{doc.page_content}")   # 修正拼写
    return "\n\n".join(parts)

# ------------------- Prompt -------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
        "你是知识库问答助手。请严格根据以下上下文回答问题。\n"
        "在回答的末尾，另起一行添加“📚 引用来源：”，然后列出本次回答用到的所有来源（用分号分隔）。\n"
        "如果上下文中没有足够信息，请明确说明。\n\n"
        "上下文：\n{context}"
    ),
    ("human", "{question}")
])

# ------------------- RAG 链 -------------------
rag_chain_with_sources = (
    RunnableParallel(
        context=retriever | format_docs_with_source,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------- 辅助函数 -------------------
def extract_sources(docs: List[Document]) -> List[str]:
    return list(set(doc.metadata.get("source", "未知") for doc in docs))

def structured_chain(query: str) -> dict:
    if not isinstance(query, str) or not query.strip():
        return {"answer": "查询无效", "sources": []}
    docs = retriever.invoke(query)
    answer = rag_chain_with_sources.invoke(query)
    sources = extract_sources(docs)
    return {"answer": answer, "sources": sources}

# ------------------- 测试 -------------------
if __name__ == "__main__":
    test_queries = [
        "大语言模型通常有多少参数？",
        "什么是向量数据库？",
        "今天天气怎么样？"
    ]
    print("=" * 60)
    print("🔗 增强版 RAG 链 (回答附带引用来源)")
    for q in test_queries:
        print(f"\n❓ 用户: {q}")
        result = structured_chain(q)
        print(f"🤖 助手: {result['answer']}")
        if result["sources"]:
            print(f"\n📂 检测到的原始来源: {', '.join(result['sources'])}")