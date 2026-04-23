# 1. 导入所需模块
import os

# 核心组件导入
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
# 2. 初始化模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
      temperature=0.7,
      openai_api_key=os.getenv("ZHIPU_API_KEY"),      # 智谱的 Key
      openai_api_base=os.getenv("ZHIPU_BASE_URL")
     )     # 智谱的端点)

# 3. 用于存储不同会话 (session) 的对话历史
# 在生产环境中，这通常是一个数据库，例如 Redis 或 SQLite
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    根据会话ID获取或创建对话历史对象。
    这是 RunnableWithMessageHistory 的核心，它将记忆的管理外部化。
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4. 定义包含历史信息的提示词模板
# 这里定义了一个系统提示词和一个占位符来接收对话历史
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手，请根据对话历史回答用户的问题。"),
        # MessagesPlaceholder 会自动被 RunnableWithMessageHistory 中的历史消息填充
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# 5. 构建 LCEL 链
# 链式调用：提示词 -> 模型
chain = prompt | llm

# 6. 用 RunnableWithMessageHistory 包装核心链
# 这是替代旧的 ConversationChain 的关键一步
conversation_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input", # 用户输入在提示词中的键名
    history_messages_key="history", # 历史消息在提示词中的键名
)

# 7. 测试多轮对话
# 注意：通过 config 参数来区分和隔离不同会话的对话历史
config = {"configurable": {"session_id": "user_123"}}

print("=== 第一轮 ===")
response1 = conversation_with_memory.invoke(
    {"input": "我叫张三，我是一名自动化测试工程师，最近在学习大模型。"},
    config=config,
)
print(f"Bot: {response1.content}")

print("\n=== 第二轮 (测试记忆) ===")
response2 = conversation_with_memory.invoke(
    {"input": "你记得我叫什么名字吗？我在做什么工作？"},
    config=config,
)
print(f"Bot: {response2.content}")

# 8. 深度原理检查点：观察内存中的历史
print("\n=== 底层 Memory 数据结构 (会话: user_123) ===")
session_history = store["user_123"]
messages = session_history.messages
print(f"消息数量: {len(messages)}")
for i, msg in enumerate(messages):
    print(f"  [{i}] {msg.type.upper()}: {msg.content[:50]}...")

# 演示新会话的独立性
print("\n=== 测试新会话 (新用户) ===")
config_new = {"configurable": {"session_id": "user_456"}}
response_new = conversation_with_memory.invoke(
    {"input": "我叫李四，我们刚刚说到哪了？"},
    config=config_new,
)
print(f"Bot to new user: {response_new.content}")
print(f"新会话历史消息数: {len(store['user_456'].messages)}")