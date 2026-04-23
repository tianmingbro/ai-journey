import os
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.callbacks import StdOutCallbackHandler
from dotenv import load_dotenv

load_dotenv()
# 1. 初始化模型 (使用较低温度以保证回答稳定性)
llm = ChatOpenAI(temperature=0.7, model=os.getenv("MODEL_NAME"))

# 2. 初始化 Memory (不限制大小)
# 注意：return_messages=True 是为了后面方便做 Token 计算
memory = ConversationBufferMemory(return_messages=True)

# 3. 构建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,  # 调试必备：观察 Prompt 是如何拼接历史的
    callbacks=[StdOutCallbackHandler()]
)

# 4. 测试多轮对话
print("=== 第一轮 ===")
response1 = conversation.predict(input="我叫张三，我是一名自动化测试工程师，最近在学习大模型。")
print(f"Bot: {response1}")

print("\n=== 第二轮 (测试记忆) ===")
response2 = conversation.predict(input="你记得我叫什么名字吗？我在做什么工作？")
print(f"Bot: {response2}")

# 5. 深度原理检查点：打印 Memory 内部存储结构
print("\n=== 底层 Memory 数据结构 ===")
print(f"Memory 类型: {type(memory.chat_memory.messages)}")
print(f"消息数量: {len(memory.chat_memory.messages)}")
for i, msg in enumerate(memory.chat_memory.messages):
    print(f"  [{i}] {msg.type.upper()}: {msg.content[:50]}...")