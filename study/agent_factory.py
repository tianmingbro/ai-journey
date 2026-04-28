import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

# 导入三件套
from study.weather_tool import get_weather
from study.calculator_tool import calculator
from study.search_tool import web_search

def build_agent(streaming: bool = False):
    """返回已配置好的三工具 Agent"""
    llm = ChatOpenAI(
        temperature=0,
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        openai_api_key=os.getenv("ZHIPU_API_KEY"),      # 智谱的 Key
        openai_api_base=os.getenv("ZHIPU_BASE_URL"),
        streaming=streaming
    )
    tools = [get_weather, calculator, web_search]
    retry_middleware = ToolRetryMiddleware(
        max_retries=2,
        backoff_factor=2.0,
        initial_delay=1.0
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是通用助手，可以查天气、算数、搜索实时信息。",
        middleware=[retry_middleware]
    )