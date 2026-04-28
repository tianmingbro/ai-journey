from api_client import QwenClient

def main():
    # 使用 QwenClient 发送请求
    with QwenClient() as client:
        messages = [
            {"role": "user", "content": "用一句话解释什么是API密钥。"}
        ]
        try:
            reply = client.chat(messages, model="qwen-turbo")
            print("通义千问:", reply)
        except Exception as e:
            print(f"调用失败: {e}")

# --- 智谱AI (ZhipuAI) ---
# from zhipuai import ZhipuAI
# client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
# response = client.chat.completions.create(
#     model="glm-4-flash", # 免费模型
#     messages=[{"role": "user", "content": "用一句话解释什么是API密钥。"}],
# )
# print("智谱AI:", response.choices[0].message.content)

# --- OpenAI ---
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "用一句话解释什么是API密钥。"}],
# )
# print("OpenAI:", response.choices[0].message.content)

if __name__ == "__main__":
    main()