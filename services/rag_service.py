from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from config.settings import settings
from utils.llm import get_llm
from services.vector_store_service import VectorStoreService

class RAGService:
    """带记忆的rag问答服务，管理检索、记忆和llm调用"""
    def __init__(self):
        # print("[RAGService] 初始化开始...")
        self._store:dict[str,ChatMessageHistory]={}
        # print("[RAGService] 加载 LLM...")
        self.llm=get_llm()
        self.stream_llm=get_llm(streaming=True)
        # print("[RAGService] LLM 加载完成，加载向量库...")
        #初始化检索器
        vs=VectorStoreService.load_existing()
        # print("[RAGService] 向量库加载完成，初始化检索器...")
        self.retriever=vs.as_retriever(search_kwargs={"k":settings.retrieval_k})

        #构建核心rag链
        common_prep=(
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self._load_history)|(lambda x:x["chat_history"]),
                context=itemgetter("question")|self.retriever|self._format_docs
            )
            |ChatPromptTemplate.from_messages([
                ("system",
                "你是一个严格基于知识库的问答助手。\n"
                "请**只**根据以下上下文和对话历史回答问题。\n"
                "如果上下文中没有足够信息，请明确回复“知识库中未找到相关信息”。\n\n"
                "特别注意：如果用户问题中包含指代词（如“它”、“那个”、“这个”），"
                "而对话历史为空或你不知道指代的是什么，你必须回复“抱歉，我不清楚您指的是什么，请提供更多上下文。\n\n"
                "上下文：\n{context}"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{question}")
            ])
        )
        print(common_prep)
        self.chain=common_prep|self.llm|StrOutputParser()
        self.stream_chain=common_prep|self.stream_llm|StrOutputParser()

    @staticmethod
    def _format_docs(docs)->str:
        """将检索到的文档列表格式化为上线文字符串"""
        return "\n\n".join(
            f"[来源：{doc.metadata.get('source','unknown')}]\n{doc.page_content}"
            for doc in docs
        )
    def _load_history(self,inputs:dict)->dict:
        """从内存字典中获取会话的聊天历史"""
        sid=inputs.get("session_id","default")
        if sid not in self._store:
            self._store[sid]=ChatMessageHistory()
        return {**inputs,"chat_history":self._store[sid].messages}
    def chat(self,session_id:str,question:str)->str:
        """执行一次问答，并自动保存本轮对话到历史"""
        history=self._store.get(session_id,ChatMessageHistory())
        answer=self.chain.invoke({
            "session_id":session_id,
            "question":question
        })
        history.add_user_message(question)
        history.add_ai_message(answer)
        self._store[session_id]=history
        return answer
    
    async def astream(self,session_id:str,question:str):
        """
        异步流式生成，逐个产出 token。
        使用 self.chain.astream() 获取每一步的更新。
        """
        #历史加载逻辑与chat（）相同
        history=self._store.get(session_id,ChatMessageHistory())
        #初始化输入
        inputs={"session_id":session_id,"question":question}
        full_answer = []  # 收集完整答案，用于最后保存
        #使用lceel的异步流式接口
        try:
            async for chunk in self.stream_chain.astream(inputs):
                #chunk是每一步的完整输出，可能包含token片段
                #对于stroutputparser之后的链，chunk直接是字符串片段
                full_answer.append(chunk)
                yield chunk
        except Exception as e:
            yield f"【出错】{e}"
        finally:
            # 保存本轮对话
            final_text = "".join(full_answer).strip()
            if final_text:
                history.add_user_message(question)
                history.add_ai_message(final_text)
                self._store[session_id] = history
    def clear_history(self,session_id:str)->None:
        """清除指定会话的聊天历史"""
        self._store.pop(session_id,None)
