import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. 定义你想要的数据结构 (Pydantic V2 风格)
class LearningPath(BaseModel):
    topic: str = Field(description="学习主题")
    steps: list[str] = Field(description="详细学习步骤列表")
    estimated_hours: int = Field(description="预估总耗时(小时)")

# 2. 初始化解析器
parser = PydanticOutputParser(pydantic_object=LearningPath)

# 3. 构建 Prompt，注意必须把 format_instructions 传给模型
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的课程规划师。请严格按照JSON格式输出结果。\n{format_instructions}"),
    ("user", "我想从零基础开始学习 {skill}，请制定一个学习路径。")
])
llm = ChatOpenAI(
    model="qwen-turbo", # 指定你想用的通义模型
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", # 通义千问兼容端点
    temperature=0.7)
# 4. 组装链
chain = prompt | ChatOpenAI(
    model="qwen-turbo", # 指定你想用的通义模型
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", # 通义千问兼容端点
    temperature=0.7) | parser

# 5. 运行
result = chain.invoke({
    "skill": "LangChain 框架",
    "format_instructions": parser.get_format_instructions()
})

print(f"主题: {result.topic}")
print(f"耗时: {result.estimated_hours}h")
for i, step in enumerate(result.steps, 1):
    print(f"步骤{i}: {step}")