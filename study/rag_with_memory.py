import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import List, Dict
print("LANGCHAIN_API_KEY loaded:", os.getenv("LANGSMITH_API_KEY")[:10] + "...")
#1.初始化llm和向量存储
llm=ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "qwen-turbo"),           # 你的对话模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),               # 统一使用 api_key 参数
    base_url=os.getenv("DASHSCOPE_BASE_URL"),             # 统一使用 base_url 参数
)
embeddings=OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    check_embedding_ctx_length=False)
vectorstore=Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="knowledge_base"
)
retriever=vectorstore.as_retriever(search_kwargs={'k':3})
def format_docs(docs:List[Document])->str:
    """将检索文档格式化为上下文字符串，附带来源"""
    return "\n\n".join(
        f"[来源：{doc.metadata.get('source','unknown')}]\n{doc.page_content}"for doc in docs
    )
#2.管理对话历史的简易存储
session_store:Dict[str,ChatMessageHistory]={}
def get_session_history(session_id:str)->ChatMessageHistory:
    """获取或创建指定session的聊天历史记录"""
    if session_id not in session_store:
        session_store[session_id]=ChatMessageHistory()
    return session_store[session_id]
#3.定义记忆注入的runnablelambda
def load_history(input_dict:dict)->dict:
    """
    从input_history中提取session_id，加载对应的聊天历史，并将其作为‘chat_history’字段合并到输入字典中"""
    session_id=input_dict.get("session_id","default")
    history=get_session_history(session_id)
    #将chatmessagehistory中的消息列表直接赋值给chat_history
    return {**input_dict,"chat_history":history.messages}
#4.构建包含历史占位符的prompt
prompt_with_history=ChatPromptTemplate.from_messages([
    ("system", 
        "你是一个智能助手，能结合对话历史和检索知识回答问题。\n"
        "如果检索到的上下文不足以回答，请如实说明。\n\n"
        "检索上下文：\n{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{question}")
])
#5.用lcel串联带记忆的rag链
conversational_rag_chain=(
    #第一步：将chat_history动态注入到输入中
    RunnablePassthrough.assign(
        chat_history=RunnableLambda(load_history)|(lambda x:x["chat_history"]),
        context=RunnableLambda(lambda x: x["question"]) | retriever | format_docs #检索依然基于原始问题
    )
    |prompt_with_history
    |llm
    |StrOutputParser
)
#6.封装问答函数，自动保存历史
# def ask_question(session_id:str,question:str)->str:
#     """执行一次问答，并自动将本轮对话保存到历史中"""
#     history=get_session_history(session_id)
#     #调用链：输入为dict，包含session_id和question
#     response=conversational_rag_chain.invoke({
#         "session_id":session_id,
#         "question":question
#     })
#     #保存本轮对话
#     history.add_user_message(question)
#     history.add_ai_message(response)
#     return response
def ask_question(session_id: str, question: str) -> str:
    history = get_session_history(session_id)
    chat_history = history.messages

    # 手动执行检索和格式化
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # 组装完整的输入字典
    input_dict = {
        "context": context,
        "chat_history": chat_history,
        "question": question
    }

    # 直接由 Prompt → LLM → 输出解析
    chain = prompt_with_history | llm | StrOutputParser()
    response = chain.invoke(input_dict)

    # 保存本轮对话
    history.add_user_message(question)
    history.add_ai_message(response)
    return response
#7.交互式测试
if __name__=="__main__":
    session_id="demo_session_001"
    print("带记忆的rag助手已就绪。（输入‘exit’退出")
    print("你可以尝试追问：'它有什么特点？' 或 '那另一个呢？'\n")
    while True:
        user_input=input("你：")
        if user_input.lower() in ["exit","quit"]:
            print("zaijian")
            break
        answer=ask_question(session_id,user_input)
        print(f"助手：{answer}\n")