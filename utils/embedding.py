from langchain_openai import OpenAIEmbeddings
from config.settings import settings

def get_embeddings():
    """返回配置好的OpenAIEmbeddings实例"""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.base_url,
        check_embedding_ctx_length=settings.check_embedding_ctx_length,
        chunk_size=settings.embedding_chunk_size,    
        )