import os
from openai import OpenAI
from dotenv import load_dotenv   # 导入加载器

# 1. 加载 .env 文件中的变量到环境变量中
load_dotenv()

# 2. 从环境变量中读取 API Key
api_key = os.getenv('ZHIPU_API_KEY')
if not api_key:
    raise ValueError("请在 .env 文件中设置 ZHIPU_API_KEY")

# 3. 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# 4. 可选：从环境变量读取模型名称（带默认值）
MODEL = os.getenv('MODEL_NAME', 'glm-4-flash')

def chat():
    print("对话助手（输入 exit 退出）")
    messages = []
    
    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() == 'exit':
            break
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            print(f"AI：{answer}")
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"错误：{e}")

if __name__ == "__main__":
    chat()