# calculator.py
import math
import re
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """安全地计算数学表达式。支持四则运算、乘方、常用科学函数。
    
    何时调用：当用户请求进行数学计算时使用此工具。
    例如：“1234*5678等于多少”、“sin(0.5)是多少”。
    
    Args:
        expression: 数学表达式字符串，只允许数字、运算符、括号、小数点和下列函数名：
                    sqrt, sin, cos, tan, log, log2, log10, pi, e, abs, ceil, floor
    Returns:
        计算结果的字符串，或在表达式不合法/计算失败时返回错误提示。
    """
    # 1. 允许的字符白名单（非字母部分）
    allowed_chars = set('0123456789+-*/()^. ,')
    # 2. 允许的函数名列表
    allowed_funcs = {
        'sqrt', 'sin', 'cos', 'tan', 'log', 'log2', 'log10',
        'pi', 'e', 'abs', 'ceil', 'floor'
    }
    
    clean = expression.strip()
    if not clean:
        return "错误：表达式为空"
    
    # 检查每一个字符
    for ch in clean:
        if ch.isalpha():
            continue  # 字母会后面单独检查函数名
        if ch not in allowed_chars:
            return f"错误：不允许的字符 '{ch}'"
    
    # 提取所有连续的字母序列，验证它们都是合法函数名
    tokens = re.findall(r'[a-zA-Z]+', clean)
    for token in tokens:
        if token.lower() not in allowed_funcs:
            return f"错误：不支持的函数或变量 '{token}'"
    
    # 构造安全的执行环境
    safe_dict = {
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
        'tan': math.tan, 'log': math.log, 'log2': math.log2,
        'log10': math.log10, 'pi': math.pi, 'e': math.e,
        'abs': abs, 'ceil': math.ceil, 'floor': math.floor
    }
    
    try:
        # __builtins__ 设为空，杜绝内置函数的访问
        result = eval(clean, {"__builtins__": {}}, safe_dict)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"


# 独立测试
if __name__ == "__main__":
    # 正常计算
    print(calculator.invoke({"expression": "3*4+5"}))
    print(calculator.invoke({"expression": "sqrt(16)"}))
    print(calculator.invoke({"expression": "log2(8)"}))
    
    # 安全测试：尝试调用危险操作
    print(calculator.invoke({"expression": "__import__('os').system('ls')"}))
    print(calculator.invoke({"expression": "open('/etc/passwd')"}))
    print(calculator.invoke({"expression": "10**10**10"}))  # 可能超时，但表达式合法（注意：未限制计算时间）