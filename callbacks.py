import logging
from langchain_core.callbacks import BaseCallbackHandler

# 配置 logging（生产环境可换成 JSON 格式 + ELK）
logger = logging.getLogger("agent.tools")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("agent_tools.log")
handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
))
logger.addHandler(handler)

class ToolLoggingCallback(BaseCallbackHandler):
    """将工具调用写入日志文件，不干扰 HTTP 响应流"""
    
    def __init__(self, session_id: str = "unknown"):
        self.session_id = session_id
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        logger.info(f"[session={self.session_id}] 🔧 {tool_name} 被调用, 参数: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"[session={self.session_id}] 📤 工具返回: {str(output)[:200]}")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"[session={self.session_id}] ❌ 工具调用失败: {error}")