import os
import uuid
import uvicorn
from dotenv import load_dotenv

from callbacks import ToolLoggingCallback
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from models import ChatRequest, ChatResponse
from agent_factory import build_agent

app = FastAPI(title="Multi-Tool Agent API")

# 创建一个同步 Agent 用于 /chat
agent = build_agent(streaming=False)
# 创建一个流式 Agent 用于 /chat/stream
stream_agent = build_agent(streaming=True)

def extract_reply_from_result(result: dict) -> str:
    """从 Agent 返回的状态中提取最后一条 AI 消息的文本"""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            return msg.content
    return "（无回复）"

def extract_tool_calls(result: dict) -> list[str]:
    """提取本次对话中调用的所有工具名"""
    tool_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc.get("name", "unknown"))
        # 有些版本可能是 ToolMessage 类型
        if hasattr(msg, "name") and msg.type == "tool":
            tool_calls.append(msg.name)
    return tool_calls
def extract_tool_calls(result: dict) -> list[str]:
    """准确提取本次对话中工具调用的名称（去重）"""
    tool_names = []
    seen_ids = set()
    for msg in result.get("messages", []):
        # 只从 AI 消息中携带的 tool_calls 提取，避免与 ToolMessage 重复
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.get("id", "")
                if call_id and call_id not in seen_ids:
                    tool_names.append(tc.get("name", "unknown"))
                    seen_ids.add(call_id)
                elif not call_id:
                    # 没有 id 的极少数情况，用名字简单去重
                    name = tc.get("name", "unknown")
                    if name not in tool_names:
                        tool_names.append(name)
    return tool_names
@app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     """标准一次性回答：Agent 完成全部思考后返回"""
#     session_id = request.session_id or str(uuid.uuid4())[:8]
#     result = agent.invoke(
#         {"messages": [{"role": "user", "content": request.message}]}
#     )
#      # === 临时调试日志 ===
#     print("\n" + "=" * 60)
#     print("完整消息历史：")
#     for i, msg in enumerate(result.get("messages", [])):
#         msg_type = getattr(msg, "type", "?")
#         content_preview = str(getattr(msg, "content", ""))[:80]
#         tool_calls = getattr(msg, "tool_calls", None)
#         msg_name = getattr(msg, "name", None)
#         print(f"  [{i}] type={msg_type}, name={msg_name}, content={content_preview}")
#         if tool_calls:
#             print(f"       tool_calls={tool_calls}")
#     print("=" * 60 + "\n")
#     # === 调试结束 ==
#     reply = extract_reply_from_result(result)
#     tools_used = extract_tool_calls(result)
#     return ChatResponse(
#         reply=reply,
#         session_id=session_id,
#         tool_calls_made=tools_used,
#         status="success"
#     )
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    # 1. 创建本次请求的 Callback 实例
    tool_logger = ToolLoggingCallback(session_id=session_id)
    
    # 2. 调用 Agent 时通过 config 传入 callbacks
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config={"callbacks": [tool_logger]}   # 👈 每次请求独立注入
    )
    
    reply = extract_reply_from_result(result)
    tools_used = extract_tool_calls(result)
    return ChatResponse(
        reply=reply,
        session_id=session_id,
        tool_calls_made=tools_used,
        status="success"
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式端点：SSE 推送 token，实时显示 Agent 思考过程"""
    session_id = request.session_id or str(uuid.uuid4())[:8]
    tool_logger = ToolLoggingCallback(session_id=session_id)

    async def event_generator():
        # 使用 astream 获取每一步的更新
        async for chunk in stream_agent.astream(
            {"messages": [{"role": "user", "content": request.message}]},
            config={"callbacks": [tool_logger]}
        ):
            # chunk 是每一步状态变化的字典，我们直接发送其 JSON 序列化
            # 前端可解析 messages 字段获取最新 token
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 配合 Nginx 使用
        }
    )

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    observer = ToolLoggingCallback()
    test_queries = [
        "计算 sin(pi/2) + sqrt(144)",             # 合法科学计算
        "帮我算 100 除以 0",                       # 除零错误（测试友好提示）
        "请计算 sqrt(-1)",                         # ValueError（测试错误反馈）
        "北京天气怎么样？然后帮我计算 （（（3+5）*2）"         # 括号不匹配（测试非法表达式）
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"用户: {query}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": [observer]}
        )
        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                print(f"Agent: {msg.content}")