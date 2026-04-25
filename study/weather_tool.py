import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。当用户询问某个城市的天气时，调用此工具。
    
    Args:
        city: 城市名称，例如 "北京"、"上海"、"Tokyo"
    
    Returns:
        该城市的天气信息字符串
    """
    # 模拟天气数据（生产环境请替换为真实 API 调用）
    weather_db = {
        "北京": "晴天，22°C，湿度 35%，北风 3 级",
        "上海": "多云，26°C，湿度 65%，东南风 2 级",
        "广州": "雷阵雨，30°C，湿度 85%，南风 4 级",
        "深圳": "阴天，28°C，湿度 70%，东风 3 级",
        "tokyo": "晴天，18°C，湿度 45%，西北风 5 级",
    }
    city_lower = city.strip().lower()
    if city_lower in weather_db:
        return f"{city}天气：{weather_db[city_lower]}"
    else:
        # 防御性设计：不给模型幻觉空间，明确返回
        return f"抱歉，暂时没有 {city} 的天气数据。请尝试查询：北京、上海、广州、深圳、Tokyo。"
# ========== 步骤3：创建 Agent ==========

# 初始化模型
llm = ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),      
    openai_api_key=os.getenv("ZHIPU_API_KEY"),      # 智谱的 Key
      openai_api_base=os.getenv("ZHIPU_BASE_URL")
)

# 系统提示词：定义 Agent 的角色和行为边界
system_prompt = """你是一个专业的天气助手。你的职责是回答用户关于天气的问题。
- 当用户询问某个城市的天气时，使用 get_weather 工具获取数据。
- 当用户只是闲聊或问其他问题时，直接礼貌回答，不要调用工具。
- 回答天气时，用自然语言转述工具返回的结果，而不是直接复制 JSON。
"""

# 一行创建 Agent：模型 + 工具 + 提示词 → 完整的 Agent 图
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=system_prompt
)

print(f"Agent 类型: {type(agent).__name__}")
# 预期输出: CompiledStateGraph (底层是 LangGraph 的有状态图)


# ========== 步骤4：测试 Agent 决策能力 ==========
if __name__ == "__main__":
    print("\n=== Agent 交互测试 ===\n")
    
    # 测试1：需要调用工具的问题
    print("【测试1】用户: 北京今天天气怎么样？")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
    })
    # 从消息列表中提取最后一条 AI 消息
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.type == "ai":
            print(f"Agent: {msg.content}")
    print("-" * 40)
    
    # 测试2：不需要工具的问题
    print("【测试2】用户: 你好，请介绍一下你自己")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]
    })
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.type == "ai":
            print(f"Agent: {msg.content}")
    print("-" * 40)
    
    # 测试3：边界情况
    print("【测试3】用户: 火星今天天气怎么样？")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "火星今天天气怎么样？"}]
    })
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.type == "ai":
            print(f"Agent: {msg.content}")