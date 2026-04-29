import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableParallel
from langchain_core.documents import Document
from typing import List

DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")   # 例如 https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 缺少 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件")
#1.复用chromamanager获取retriever
class ChromaManager:
    """封装chroma的常用查询和管理操作"""
    def __init__(self,persist_directory="./chroma_db",collection_name="knowledge_base"):
        self.embeddings=OpenAIEmbeddings(
            model="text-embedding-v3",
             base_url=DASHSCOPE_BASE_URL,      # 关键：指向阿里云
            api_key=DASHSCOPE_API_KEY,
            check_embedding_ctx_length=False
        )
        self.vectorstore=Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    def as_retriever(self,k=3):
        return self.vectorstore.as_retriever(search_kwargs={"k":k})
        
#初始化管理器并获取检索器
manager=ChromaManager()
retriever=manager.as_retriever(k=3)
#2.初始化llm
llm=ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME","gpt-4.1-mini"),
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),      # 智谱的 Key
    openai_api_base=os.getenv("DASHSCOPE_BASE_URL")
)
#3.文档格式化函数
def format_docs(docs:List[Document])->str:
    """将检索到的文档列表拼接位一个上下文字符串"""
    return "\n\n".join(
        f"[来源{i+1}]{doc.page_content}" for i, doc in enumerate(docs)
    )
#4.prompt模板
prompt=ChatPromptTemplate.from_messages([
    ("system",
        "你是一个基于知识库的问答助手。请严格根据以下上下文回答问题。\n"
        "如果上下文中没有足够信息，请明确说'知识库中未找到相关信息'。\n\n"
        "上下文：\n{context}"),
        ("human","{question}")
])
#5.用lcel管道串联rag全流程
rag_chain=(
    RunnableParallel(context=retriever | format_docs,
    question=RunnablePassthrough()
    )
    |prompt
    |llm
    |StrOutputParser()
)
#6.测试
if __name__=="__main__":
    test_queries=[
                "大语言模型通常有多少参数？",
        "什么是向量数据库？",
        "Transformer架构的核心是什么？",
        "今天天气怎么样？"
    ]
    print("rag链测试")
    print(f"知识库文档数量：{manager.vectorstore._collection.count()}")
    for query in test_queries:
        print(f"\n")
        print(f"用户:{query}")
        try:
            answer=rag_chain.invoke(query)
            print(f"助手：{answer}")
        except Exception as e:
            print(f"调用出错：{e}")