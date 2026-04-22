import os
from dotenv import load_dotenv

load_dotenv()

--- 通义千问 (DashScope) ---
from dashscope import Generation
response = Generation.call(
    model="qwen-turbo",
    messages=[{"role": "user", "content": "用一句话解释什么是API密钥。"}],
    api_key=os.getenv("DASHSCOPE_API_KEY")
)
print("通义千问:", response.output["text"])

--- 智谱AI (ZhipuAI) ---
from zhipuai import ZhipuAI
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
response = client.chat.completions.create(
    model="glm-4-flash", # 免费模型
    messages=[{"role": "user", "content": "用一句话解释什么是API密钥。"}],
)
print("智谱AI:", response.choices[0].message.content)

# --- OpenAI ---
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "用一句话解释什么是API密钥。"}],
# )
# print("OpenAI:", response.choices[0].message.content)

