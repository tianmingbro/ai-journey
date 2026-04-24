from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

# 链 1: 专门用于扩写提示词的专家
expander_prompt = PromptTemplate.from_template(
    "你是一位提示词工程专家。请将用户的简短需求，扩写成一个结构清晰、包含示例要求的高质量提示词。"
    "用户需求: {user_input}\n"
    "高质量提示词:"
)
expander_chain = expander_prompt | llm | StrOutputParser()

# 链 2: 专门执行被优化后提示词的 AI
executor_chain = PromptTemplate.from_template("{optimized_prompt}") | llm | StrOutputParser()

# 业务逻辑串联
user_raw_input = "帮我写一个计算斐波那契数列的Python函数"

print("🧠 正在优化提示词...")
optimized_prompt = expander_chain.invoke({"user_input": user_raw_input})
print(f"✨ 优化后的提示词为:\n{optimized_prompt}\n")

print("🤖 正在生成回答...")
final_answer = executor_chain.invoke({"optimized_prompt": optimized_prompt})
print(f"📝 最终结果:\n{final_answer}")