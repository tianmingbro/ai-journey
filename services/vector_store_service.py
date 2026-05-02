import os
import time
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings
from utils.embedding import get_embeddings

class VectorStoreService:
    """封装Chroma向量数据库的创建于加载"""

    @staticmethod
    def create_from_documents(docs:List[Document])->Chroma:
        """从文档列表创建新的向量存储（覆盖已有数据）"""
        # 先创建空的 Chroma 实例（持久化目录会自动创建）
        vectorstore = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # 然后批量添加文档（内部会处理 batch，避免参数冲突）
        # 分批添加文档，每批最多 10 条（OpenAI 限制）
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            vectorstore.add_documents(batch)    
            if i + batch_size < len(docs):
                time.sleep(0.5)    
        return vectorstore
    
    @staticmethod
    def add_documents(docs: List[Document]):
        """
        向现有知识库追加文档。如果知识库不存在，则创建。
        """
        vs = VectorStoreService.load_existing()
        if vs is None:
            # 知识库还不存在，就创建一个新的（保留 create_from_documents 用于初始建库）
            VectorStoreService.create_from_documents(docs)
        else:
            vs.add_documents(docs)
    @staticmethod
    def load_existing() -> Optional[Chroma]:        
    # def load_existing()->Chroma:
        """加载已持久化的向量存储"""
        persist_dir = settings.chroma_persist_dir
        # if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        #     raise FileNotFoundError(
        #         f"向量库目录 '{persist_dir}' 不存在或为空。请先运行 DocumentService 和 "
        #         "VectorStoreService.create_from_documents() 初始化知识库。"
        #     )
        # return Chroma(
        #     embedding_function=get_embeddings(),
        #     persist_directory=persist_dir,
        #     collection_name=settings.collection_name
        # )
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            return None
        try:
            return Chroma(
                embedding_function=get_embeddings(),
                persist_directory=persist_dir,
                collection_name=settings.collection_name
            )
        except Exception:
            return None