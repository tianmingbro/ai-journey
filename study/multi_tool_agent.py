import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.callbacks import BaseCallbackHandler

from study.weather_tool import get_weather
from study.calculator_tool import calculator
from study.search_tool import web_search

# ========== 初始化模型 ==========
llm = ChatOpenAI(temperature=0, model=os.getenv("MODEL_NAME", "gpt-4.1-mini"))

# ========== 工具三件套 ==========
tools = [get_weather, calculator, web_search]

# ========== ToolRetryMiddleware ==========
# 第1次失败后等待 2s，第2次等待 4s，第3次等待 8s
tool_retry = ToolRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=30.0
    # retry_on=(requests.Timeout, requests.HTTPError)  # 可指定只重试特定异常
)

# ========== 自定义回调：观察工具选择 ==========
class ToolObserver(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  🔧 Agent 选择调用: {serialized.get('name', 'unknown')}")
        print(f"     📥 参数: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"     📤 结果: {str(output)[:80]}...")

# ========== 创建 Agent ==========
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一个通用助手，可以完成以下三类任务：
1. 天气查询：用户询问某个城市的天气时，使用 get_weather 工具。
2. 数学计算：用户需要计算时，使用 calculator 工具。
3. 实时信息搜索：用户询问最新资讯、百科知识或需要联网搜索时，使用搜索工具。
请根据用户请求选择正确的工具。能用一个工具回答就不要同时用多个。""",
    middleware=[tool_retry]  # 👈 注入重试中间件
)

# ========== 测试 ==========
if __name__ == "__main__":
    observer = ToolObserver()
    test_cases = [
        "北京今天天气怎么样？",          # → get_weather
        "计算 1234 乘以 5678 等于多少？", # → calculator
        "最近大语言模型有什么最新进展？"   # → search
    ]
    
    for query in test_cases:
        print(f"\n{'='*50}")
        print(f"用户: {query}")
        print(f"{'-'*50}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": [observer]}
        )
        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                print(f"助手: {msg.content}")