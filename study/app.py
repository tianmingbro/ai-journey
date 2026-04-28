import uvicorn
from fastapi import FastAPI
from study.models import ChatRequest, ChatResponse

app = FastAPI(
    title="Echo 助手",
    description="一个演示 Pydantic 校验与 Swagger UI 的最小服务",
    version="0.1.0"
)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """接收消息并原样返回，同时统计长度"""
    session_id = request.session_id or "auto-generated-session"
    return ChatResponse(
        reply=f"Echo: {request.message}",
        session_id=session_id,
        received_length=len(request.message),
        status="success"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)