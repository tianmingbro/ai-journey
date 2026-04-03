import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('ZHIPU_API_KEY'),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

def chat():
    print("智谱AI对话助手（输入 exit 退出）")
    messages = []
    
    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() == 'exit':
            print("再见！")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",  # 永久免费模型
                messages=messages,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            print(f"AI：{answer}")
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"调用出错：{e}")

if __name__ == "__main__":
    chat()