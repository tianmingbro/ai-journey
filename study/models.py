# from pydantic import BaseModel, Field
# from typing import Optional, List, Literal

# class ChatRequest(BaseModel):
#     """客户端发送的请求体"""
#     message: str = Field(
#         ...,
#         min_length=1,
#         max_length=200,
#         description="用户输入消息，不能为空且不能超过200字符"
#     )
#     session_id: Optional[str] = Field(
#         default=None,
#         pattern=r'^[a-zA-Z0-9_-]{1,36}$',
#         description="会话ID，仅允许字母数字、连字符和下划线"
#     )
#     # strict 模式示例：如果需要严格类型，可加 strict=True
#     # age: int = Field(..., strict=True, ge=0, le=150)

#     model_config = {
#         "json_schema_extra": {
#             "examples": [
#                 {"message": "你好", "session_id": "abc-123"}
#             ]
#         }
#     }

# class ChatResponse(BaseModel):
#     """服务端返回的响应体"""
#     reply: str = Field(..., description="助手回复")
#     session_id: str = Field(..., description="实际使用的会话ID")
#     received_length: int = Field(..., description="收到消息的长度，方便调试")
#     status: Literal["success", "error"] = Field(default="success")
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, pattern=r'^[a-zA-Z0-9_-]{1,36}$')

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "北京天气？", "session_id": "user-001"}]
        }
    }

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls_made: List[str] = Field(default_factory=list)
    status: Literal["success", "error"] = "success"