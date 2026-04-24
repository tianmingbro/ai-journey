import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
# 初始化 LLM
llm = ChatOpenAI(temperature=0, model=os.getenv("MODEL_NAME"),openai_api_key=os.getenv("ZHIPU_API_KEY"),      # 智谱的 Key
      openai_api_base=os.getenv("ZHIPU_BASE_URL"))

# ---- 定义两个独立的步骤 ----

# 步骤1：中文转英文
translate_prompt = ChatPromptTemplate.from_template(
    "将以下中文翻译成英文。只输出英文，不要解释。\n中文：{input}"
)
translate_chain = translate_prompt | llm | StrOutputParser()

# 步骤2：英文问答
qa_prompt = ChatPromptTemplate.from_template(
    "根据以下英文内容，回答问题。\n英文内容：{input}\n问题：这段话想表达什么核心思想？请用中文回答。"
)
qa_chain = qa_prompt | llm | StrOutputParser()

# ---- 用 | 串联两个链 ----
# 注意：translate_chain 的最终输出是字符串（因为 StrOutputParser），
# 这个字符串会被自动包装成字典 {"input": 翻译好的英文}，然后传给 qa_chain。
full_chain = translate_chain | qa_chain

# ---- 测试 ----
user_input = "大语言模型正在改变我们与计算机交互的方式，它让机器能够理解并生成自然语言。"
print("=" * 60)
print(f"用户输入: {user_input}")
print("-" * 60)

result = full_chain.invoke({"input": user_input})
print(f"最终回答: {result}")

from langchain_core.runnables import RunnableLambda

# 在 translate_chain 之后截断
def print_intermediate(data):
    print("\n>>> [拦截] 翻译后的数据包内容:")
    print(f"    类型: {type(data)}")
    print(f"    内容: {data}")
    print("-" * 60)
    return data  # 必须原样返回，否则管道断裂

# 新链：翻译 → 打印包 → 回答
debug_chain = translate_chain | RunnableLambda(print_intermediate) | qa_chain

# 替代 full_chain 运行
result = debug_chain.invoke({"input": user_input})
print(f"最终回答: {result}")