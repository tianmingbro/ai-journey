import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

#1.初始化Embedding模型
embeddings=OpenAIEmbeddings(
model="text-embedding-v3",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False)
#2.构建示例文档（每条文档包含文本和元数据）
documents=[
    Document(
        page_content="大语言模型通常包含数十亿甚至数千亿参数，需要海量数据和计算资源进行训练。",
        metadata={"source":"weather_report","topic":"weather"},
        id="doc_001"
                ),
    Document(
        page_content="今天北京天气晴朗，最高气温 22 摄氏度，适合户外运动和郊游。",
        metadata={"source": "weather_report", "topic": "weather"},
        id="doc_002"
    ),
    Document(
        page_content="向量数据库是一种专门用于存储和检索高维向量的数据库系统，常用于语义搜索和推荐系统。",
        metadata={"source": "vector_db_intro", "topic": "database"},
        id="doc_003"
    ),
    Document(
        page_content="Transformer 架构中的自注意力机制是 LLM 的核心组件，由 Vaswani 等人在 2017 年提出。",
        metadata={"source": "transformer_paper", "topic": "AI"},
        id="doc_004"
    ),
    Document(
        page_content="Python 是数据科学和机器学习领域最流行的编程语言之一，拥有丰富的生态库。",
        metadata={"source": "python_intro", "topic": "programming"},
        id="doc_005"
    ),
    ]
#3.创建持久化向量存储
vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="knowledge_base",
    collection_metadata={"hnsw:space":"cosine"},
)
#4.输出创建结果
print("持久化向量存储创建成功！")
print(f"    路径： ./chroma_db/")
print(f"    集合名称：knowledge_base")
print(f"    文档数量：{len(documents)}")
print(f"    已索引的文档ID:{[doc.id for doc in documents]}")
print(f"    集合内记录总数：{vectorstore._collection.count()}")