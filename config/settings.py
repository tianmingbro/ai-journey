from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    #---运行环境配置（来自.env/环境变量）---
    openai_api_key:str
    langsmith_api_key:str=""

    # 通用模型与 API 参数
    model_name: str="qwen-plus"
    embedding_model:str="text-embedding-v3"
    base_url: Optional[str] = None                # 自定义 LLM 端点
    embedding_chunk_size: int = 10           # 批次大小
    check_embedding_ctx_length: bool = False # 是否检查 token 长度
    
    # ========== LangSmith 追踪 ==========
    langchain_tracing_v2: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    langsmith_endpoint: Optional[str] = None

     # ========== RAG 业务参数 ==========
    chroma_persist_dir: str="./chroma_db"
    collection_name:str="knowledge_base"
    chunk_size:int=400
    chunk_overlap:int=80
    retrieval_k:int=3
    temperature:float=0.0
    model_config={
        "env_file":".env",
        "env_file_encoding":"utf-8",
        "extra":"allow"
    }

#全局单例
settings=Settings()