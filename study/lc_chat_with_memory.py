import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# --- 初始化 LLM ---
llm = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo")
)

# --- 初始化 Memory（与 Day9 完全一致） ---
memory = ConversationBufferMemory(
    return_messages=True,     # 返回 Message 对象列表，方便填入 MessagesPlaceholder
    memory_key="history"      # 在 prompt 模板中占位符变量名
)

# --- Prompt 模板：使用 MessagesPlaceholder 容纳历史列表 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，请用简洁的方式回答问题。"),
    MessagesPlaceholder(variable_name="history"),   # 这里会插入历史消息列表
    ("human", "{input}")
])

# --- 动态提取记忆的函数 ---
def load_history(_):
    """从 memory 对象中加载当前对话历史，返回消息列表"""
    return memory.load_memory_variables({})["history"]

# --- 构建纯 LCEL 链 ---
chain = (
    RunnablePassthrough.assign(history=load_history) 
    | RunnableLambda(lambda x: (print("字典内容:", x.keys(), "历史消息数:", len(x["history"])), x)[1]) # 注入 history
    | prompt
    | llm
)

# --- 交互式对话循环 ---
if __name__ == "__main__":
    print("💬 带记忆的对话机器人（LCEL 原生实现）")
    print("输入 'exit' 或 'quit' 结束对话\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束。记忆已清空（未持久化）。")
            break

        # 调用链（此时链内自动读取了 memory 中的历史）
        response = chain.invoke({"input": user_input})
        ai_text = response.content

        print(f"Bot: {ai_text}")

        # 关键步骤：将本轮对话保存到 memory（副作用在链外完成）
        memory.save_context(
            {"input": user_input},
            {"output": ai_text}
        )