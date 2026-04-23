# chat_with_memory.py
import os
import tiktoken
from api_client import QwenClient, ZhipuClient  # 复用之前封装的客户端
from dotenv import load_dotenv

load_dotenv()

class ConversationMemory:
    """带 token 限制的对话记忆管理器"""
    
    def __init__(self, max_tokens: int = 2000, model: str = os.getenv("MODEL_NAME")):
        self.messages = []
        self.max_tokens = max_tokens
        self.model = model
        self.encoding = self._get_encoding()
    
    def _get_encoding(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, messages: list) -> int:
        """计算消息列表的总 token 数"""
        num_tokens = 0
        for msg in messages:
            # 每条消息格式开销（角色标识）
            num_tokens += 4
            for key, value in msg.items():
                num_tokens += len(self.encoding.encode(value))
        num_tokens += 2  # 整体回复引导符
        return num_tokens
    
    def _truncate(self):
        """若总 token 数超限，从最早的消息开始删除，保留最后一条用户消息"""
        while len(self.messages) > 1 and self._count_tokens(self.messages) > self.max_tokens:
            # 如果第一条是 assistant，可以删；但如果第一条是 user 且是最后一条 user，则保留
            if len(self.messages) <= 2:
                break
            self.messages.pop(0)
    
    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._truncate()
    
    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._truncate()
    
    def get_messages(self) -> list:
        return self.messages
    
    def clear(self):
        self.messages = []

class ChatSession:
    def __init__(self, client, memory: ConversationMemory, model: str = os.getenv("MODEL_NAME")):
        self.client = client
        self.memory = memory
        self.model = model
    
    def send(self, user_input: str) -> str:
        self.memory.add_user_message(user_input)
        messages = self.memory.get_messages()
        reply = self.client.chat(messages, model=self.model)
        self.memory.add_assistant_message(reply)
        return reply

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 初始化客户端（通义或智谱）
    client = QwenClient()  # 或 ZhipuClient()
    memory = ConversationMemory(max_tokens=500)
    session = ChatSession(client, memory)
    
    print("🤖 多轮对话已启动（输入 'exit' 退出，'clear' 清空记忆）")
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            memory.clear()
            print("🧹 记忆已清空")
            continue
        
        reply = session.send(user_input)
        print(f"🤖 助手: {reply}")
        print(f"    [当前记忆 token 数: {memory._count_tokens(memory.messages)}]")