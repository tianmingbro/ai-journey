"""def robust_api_call(url,params=None,max_retries=3,backoff=1.0):
    for attempt in range(max_retries):
        try:
            resp=requests.get(url,params,timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            if resp.status_code in (429,500,502,503,504):
                pass
            else:
                return f"api 返回错误（{resp.status_code}),不重试"
        except (requests.ConnectionError,requests.Timeout):
            pass
        time.sleep(backoff*(2**attempt))
    return "api不可用，请稍后重试"""

"""import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
#初始化llm
llm=ChatOpenAI(temperature=0,model=os.getenv("MODEL_NAME"),openai_api_key=os.open_api_base=os.getenv("ZHIPU_BASE_URL"))

#定义两个独立的步骤
#中文转英文j
translate_prompt=ChatPromptTemplate.from_template("将以下中文翻译成英文，不要解释。\n中文：{input}")
translate_chain=translate_prompt | llm |StrOutputParser()
#步骤二英文问答
qa_promt=ChatPromptTemplate.from_template("请根据以下英文回答问题。\n英文内容：{input}\n问题：这段话想表达什么核心思想？请用中文回答。")
qa_chain=qa_promt |llm |StrOutputParser
fullchain=translate_chain | qa_chain

userinput="大语言模型"
result=fullchain.invoke({"input":userinput})"""


"""from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store={}
def get_session_history(sessionid:str) -> ChatMessageHistory:
    if sessionid not in store:
        store[sessionid]=ChatMessageHistory()
    return store[sessionid]
chat_with_memory=RunnableWithMessageHistory(
    prompt | llm |StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

from langgraph.checkpoint.memory import InMemorySaver
agent=create_agent(
    model=llm,
    tools=[get_weather,calculator],
    checkpointer=InMemorySaver()
)
result=agent.invoke(
    {"messages":[{"role":"user","content":"北京天气？"}]},
    config={"configurable":{"thread_id":"user-001"}}
)
result=agent.invoke(
    {"messages":[{"role":"user","content":"那上海呢？"}]},
    config={"configurable":{"thread_id":"user-001"}}
)"""


