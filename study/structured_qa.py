import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from pydantic import BaseModel, Field

# 昨天的数据模型
class CustomerQuery(BaseModel):
    order_id: str = Field(description="用户提及的订单号")
    intent: str = Field(description="用户意图，如：查询物流、退款、投诉")

# 使用结构化输出，注意显式指定 method="json_schema" 保持与 v0.3 默认行为一致
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
      temperature=0,
      openai_api_key=os.getenv("ZHIPU_API_KEY"),      # 智谱的 Key
      openai_api_base=os.getenv("ZHIPU_BASE_URL")
     )     # 智谱的端点)
structured_llm = llm.with_structured_output(CustomerQuery, method="json_schema")

prompt = ChatPromptTemplate.from_messages([
    ("system", "提取用户意图和订单号，输出 JSON。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | structured_llm

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 测试
config = {"configurable": {"session_id": "test_structured"}}

# 第一轮
resp1 = conversation.invoke({"input": "我的订单号是 12345，我想查询物流。"}, config=config)
print(resp1)  # 应输出 order_id='12345', intent='查询物流'

# 第二轮：测试记忆下的结构化输出
resp2 = conversation.invoke({"input": "刚才那个订单号是多少？我要投诉。"}, config=config)
print(resp2)  # 期望输出 order_id='12345', intent='投诉'