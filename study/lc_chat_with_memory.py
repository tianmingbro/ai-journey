from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory # 社区记忆模块
import uuid

llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# 1. 初始化 Memory
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# 2. Prompt 模板（包含历史占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 3. 核心 LCEL 链：利用 RunnablePassthrough.assign 动态注入历史
def load_history(_):
    """从 memory 加载历史消息列表"""
    return memory.load_memory_variables({})["history"]

# 构建链
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(load_history)  # 运行时加载历史
    )
    | prompt
    | llm
)

# 4. 对话循环（模拟交互）
session_id = str(uuid.uuid4())[:8]
print(f"会话 ID: {session_id}\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # 调用链获取回复
    response = chain.invoke({"input": user_input})
    ai_msg = response.content
    
    # 将本轮对话存入 memory
    memory.save_context({"input": user_input}, {"output": ai_msg})
    # 注意：如果 prompt 里 AI 的角色是 assistant，保存时需用 "output" 键
    # memory.chat_memory.add_message(AIMessage(content=ai_msg))
    
    print(f"Bot: {ai_msg}\n")