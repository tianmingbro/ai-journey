import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# 初始化模型（沿用你昨天的配置）
llm = ChatOpenAI(
    temperature=0,
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    openai_api_key=os.getenv("ZHIPU_API_KEY"),      # 智谱的 Key
    openai_api_base=os.getenv("ZHIPU_BASE_URL")
)

# ---- 翻译支路 ----
translate_prompt = ChatPromptTemplate.from_template(
    "将以下中文翻译成英文，只输出英文，不要解释：\n{input}"
)
translate_chain = translate_prompt | llm | StrOutputParser()

# ---- 最终回答的 Prompt（注意：它需要两个变量） ----
final_prompt = ChatPromptTemplate.from_template(
    """你是一个双语助手。下面有用户的原始中文输入，以及它的英文翻译。
请用中文复述用户的问题，然后用英文给出基于该问题的回答。

原文：{original_text}
英文翻译：{english_text}

回答："""
)

# ---- 构建并行分支 ----
branch = RunnableParallel(
    english_text=translate_chain,          # 分支1：执行翻译，结果存入 english_text 键
    original_text=RunnablePassthrough()    # 分支2：原封不动传递整个输入，存入 original_text 键
)

# ---- 调试中间数据的拦截器 ----
def print_branch_output(data):
    print("\n>>> [并行分支输出] 下游即将收到的完整字典:")
    import json
    # 打印时注意 original_text 可能是嵌套的 dict
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    print("-" * 50)
    return data

# ---- 组装总链 ----
chain = (
    branch                       # 1. 分叉执行
    | RunnableLambda(print_branch_output)  # 2. 观察数据包
    | final_prompt               # 3. 填入 Prompt 模板
    | llm                        # 4. LLM 生成回答
    | StrOutputParser()          # 5. 解析成纯字符串
)

# ---- 测试 ----
if __name__ == "__main__":
    user_input = "你好，请问图书馆在哪里？"
    print("=" * 60)
    print(f"用户输入: {user_input}")
    result = chain.invoke({"input": user_input})
    print("\n" + "=" * 60)
    print(f"最终回答:\n{result}")