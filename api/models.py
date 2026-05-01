import re
from pydantic import BaseModel,Field,field_validator
from typing import Optional

#chat请求
class ChatRequest(BaseModel):
    question:str=Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户提出的问题，不能为空切不超过2000字符"

    )
    session_id:Optional[str]=Field(
        "default",
        description="会话id，用于管理多轮对话记忆。只允许字母、数字、下划线、连字符，长度1-36"
    )
    #自定义校验：session_id必须符合安全字符集
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls,v:str)->str:
        if not re.match(r'^[a-zA-Z0-9_-]{1,36}$',v):
            raise ValueError("session_id只能包含字母、数字、下划线与连字符，长度1-36")
        return v
    
    #swagger示例
    model_config={
        "json_schema_extra":{
            "examples":[
                {
                    "question":"大语言模型有多少参数？",
                    "session_id":"demo-session"
                }
            ]
        }
    }
#chat响应
class ChatResponse(BaseModel):
    answer:str=Field(...,description="模型生成的回答")
    session_id:str=Field(...,description="当前会话id")

#上传响应
class UploadResponse(BaseModel):
    filename:str=Field(...,description="上传的文件名")
    chunks:int=Field(...,description="分割后的文本块数量")
    status:str=Field("success",description="处理状态，成功是为’success‘")
    detail:str=Field("",description="附加信息，如警告或错误详情")