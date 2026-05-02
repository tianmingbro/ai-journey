import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models import ChatRequest,ChatResponse
from services.rag_service import RAGService

router=APIRouter()
rag=RAGService()    #单例，内部管理检索链和对话历史

#1.一次性问答（非流式）
@router.post("/chat",response_model=ChatResponse)
def chat(request:ChatRequest):
    """
    标准问答：收到完整问题，返回完整答案。
    内部调用 RAGService.chat，自动保存对话历史。
    """
    answer=rag.chat(request.session_id,request.question)
    return ChatResponse(
        answer=answer,
        session_id=request.session_id
    )

#2.流式回答（sse)
@router.post("/chat/stream",response_class=StreamingResponse)
async def chat_stream(request:ChatRequest):
    """
    流式问答：以 Server-Sent Events 格式逐 token 推送。
    前端可使用 EventSource 接收。
    """
    async def envent_generator():
        try:
            #ragservice的astream方法应返回异步生成器，逐个产出token
            async for token in rag.astream(request.session_id,request.question):
                #sse格式：data：内容\n\n
                yield f"data:{token}\n\n"
                # #允许事件循环切换，防止阻塞
                # await asyncio.sleep(0)
        except Exception as e:
            #流内异常也通过sse返回
            yield f"event:error\ndata:{str(e)}\n\n"
        finally:
            yield "data:[DONE]\n\n"
        
    return StreamingResponse(
        envent_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":"no-cache",
            "Connection":"keep-alive",
            "X-Accel-Buffering":"no",# 配合 Nginx 反向代理时关闭缓冲
        }
    )