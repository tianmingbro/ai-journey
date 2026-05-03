# api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import router as chat_router
from api.upload import router as upload_router
from utils.logger import get_eval_logger
from utils.logger import get_eval_logger
logger = get_eval_logger()
logger.log_eval_trace(
    question="test",
    answer="test",
    retrieved_docs=[],
    latency_ms=0,
    token_usage={}
)

app = FastAPI(
    title="RAG 问答 API",
    description="基于 LangChain 1.2 的检索增强生成问答系统，支持文件上传与多轮对话",
    version="1.0.0"
)
get_eval_logger()   # 实例化日志单例，创建 .log/ 目录和当日文件
# 跨域配置（允许前端任意来源访问，实际部署时需限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)     # /chat, /chat/stream
app.include_router(upload_router)    # /upload

# 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}

