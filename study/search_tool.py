# search.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from typing import Optional

# TavilySearchResults 本身就是一个 BaseTool，但它的默认描述是英文且比较宽泛。
# 我们可以再次用 @tool 包装，以便自定义更精准的中文描述和名称。
# 注意：需要设置环境变量 TAVILY_API_KEY
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    print("警告: 未设置 TAVILY_API_KEY 环境变量，搜索工具将无法工作。请访问 https://tavily.com 获取免费API密钥。")
    # 不退出，让后续调用时自然报错，方便测试

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """网络搜索工具，用于查询实时信息、百科知识或最新资讯。
    
    何时调用：当用户问题涉及“最新”、“今天”、“近期”等时效性要求，
    或询问超出你知识截止日期的事件，或要求查找具体网页资料时使用。
    
    Args:
        query: 搜索关键词或问句
        max_results: 返回结果的最大数量，默认为3
    Returns:
        搜索结果摘要，包含标题、链接和内容片段
    """
    # 使用 Tavily 社区工具进行实际搜索
    # 注意：TavilySearchResults 默认 max_results=5，这里可根据参数调整
    tavily_tool = TavilySearchResults(
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
        include_domains=None,
    )
    try:
        result = tavily_tool.invoke({"query": query})
        # result 往往是一个列表，包含字典；将其格式化为易读字符串
        if isinstance(result, list):
            output = f"搜索 '{query}' 的结果：\n"
            for i, item in enumerate(result, 1):
                title = item.get("title", "无标题")
                url = item.get("url", "无链接")
                content = item.get("content", "无内容")
                output += f"{i}. {title}\n   {url}\n   {content}\n"
            return output
        return str(result)
    except Exception as e:
        return f"搜索过程中出错: {e}"


# 独立测试（需要先设置 TAVILY_API_KEY 环境变量）
if __name__ == "__main__":
    print("工具名称:", web_search.name)
    print("工具描述:", web_search.description)
    print("\n测试搜索：最近大语言模型有什么最新进展？")
    res = web_search.invoke({"query": "大语言模型最新进展"})
    print(res)