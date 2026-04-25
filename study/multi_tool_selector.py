from langchain.agents.middleware import LLMToolSelectorMiddleware

# 模拟 10+ 工具的大注册表
all_tools = [
    get_weather, calculator, search,
    # ... 其他 7+ 工具（模拟大工具集）
]

# 使用小模型预筛选：每次调用前先筛选出 3 个最相关的工具
tool_selector = LLMToolSelectorMiddleware(
    tool_selector_model=ChatOpenAI(model="gpt-4.1-mini", temperature=0),
    max_tools=3,
    # prompt 可自定义筛选逻辑
)

agent_with_selector = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt="你是通用助手，根据用户问题选择合适的工具。",
    middleware=[tool_selector, tool_retry]  # 先筛选再重试
)

# 测试：工具注册表有 10+ 工具，但 LLM 只收到 3 个最相关的
result = agent_with_selector.invoke(
    {"messages": [{"role": "user", "content": "今天深圳股市怎么样？"}]}
)