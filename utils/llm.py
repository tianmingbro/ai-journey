from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm(streaming:bool=False):
    """返回配置好的ChatOpenAI实例"""
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        streaming=streaming,
        api_key=settings.openai_api_key,
        base_url=settings.base_url,
        request_timeout=30
    )