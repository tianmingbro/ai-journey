import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    check_embedding_ctx_length=False)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="knowledge_base"
)

data = vectorstore.get()
for i, doc_id in enumerate(data["ids"]):
    print(f"ID: {doc_id}")
    print(f"内容: {data['documents'][i][:100]}...")
    print(f"元数据: {data['metadatas'][i]}")
    print("-" * 50)