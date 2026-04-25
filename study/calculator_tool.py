import math
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# ============================================================
# 步骤1：Pydantic args_schema —— 参数校验的第一道防线
# ============================================================
class CalculatorInput(BaseModel):
    """计算器工具的输入 Schema——在函数执行前自动校验"""
    expression: str = Field(
        ...,
        min_length=1,
        max_length=500,
        # pattern=r'^[0-9+\-*/().%\s,^\w]+$',  # 只允许数学字符和字母（函数名）
        description="数学表达式字符串。支持四则运算、幂运算（** 或 ^）、三角函数（sin/cos/tan）、对数（log/log2/log10）、平方根（sqrt）、绝对值（abs）等。例如：'3*4+5'、'sqrt(16)*2'、'sin(pi/2)'"
    )
    precision: Optional[int] = Field(
        default=6,
        ge=1,
        le=12,
        description="计算结果的小数精度，默认 6 位"
    )


# ============================================================
# 步骤2：安全函数命名空间（Hive 风格命名 + 自由数学计算）
# ============================================================
def _build_safe_namespace() -> dict:
    """构建安全的数学命名空间，只暴露纯数学函数，杜绝任意代码执行。
    支持高阶函数如 lambdify(expr).subs(x, 3.14) 等，提高计算灵活性。
    """
    safe_dict = {
        # 基础函数
        "sqrt":   math.sqrt,
        "abs":    abs,
        "round":  round,
        "min":    min,
        "max":    max,
        "sum":    sum,
        # 三角函数
        "sin":    math.sin,
        "cos":    math.cos,
        "tan":    math.tan,
        "asin":   math.asin,
        "acos":   math.acos,
        "atan":   math.atan,
        "atan2":  math.atan2,
        # 双曲函数
        "sinh":   math.sinh,
        "cosh":   math.cosh,
        "tanh":   math.tanh,
        # 对数与指数
        "log":    math.log,
        "log2":   math.log2,
        "log10":  math.log10,
        "exp":    math.exp,
        "pow":    pow,
        # 常数
        "pi":     math.pi,
        "e":      math.e,
        "tau":    math.tau,
        "inf":    math.inf,
        # 数值处理
        "ceil":   math.ceil,
        "floor":  math.floor,
        "trunc":  math.trunc,
        "degrees": math.degrees,
        "radians": math.radians,
        # 高等函数
        "erf":    math.erf,
        "erfc":   math.erfc,
        "gamma":  math.gamma,
        "lgamma": math.lgamma,
    }
    return safe_dict


# ============================================================
# 步骤3：正则安全校验（字符级白名单过滤）
# ============================================================
def _validate_expression(expr: str) -> tuple[bool, str]:
    """正则安全校验：只允许白名单字符通过，拒绝任何可疑输入
    
    返回 (是否合法, 错误消息)。
    """
    # 白名单：数字、基本运算符、小数点、括号、空格、字母（函数名）
    ALLOWED_PATTERN = re.compile(r'^[0-9+\-*/().%\s,^a-zA-Z_]+$')
    
    if not ALLOWED_PATTERN.match(expr):
        illegal = set(re.findall(r'[^0-9+\-*/().%\s,^a-zA-Z_]', expr))
        return False, f"表达式包含非法字符：{illegal}"
    
    if not expr.strip():
        return False, "表达式不能为空"
    
    # 额外安全检查：禁止潜在的危险模式
    DANGEROUS_KEYWORDS = ['__', 'import', 'exec', 'eval', 'compile', 'open', 
                          'file', 'input', 'os', 'sys', 'subprocess', 'lambda']
    expr_lower = expr.lower()
    for kw in DANGEROUS_KEYWORDS:
        if kw in expr_lower:
            return False, f"表达式包含禁止的关键字：'{kw}'"
    
    return True, ""


# ============================================================
# 步骤4：@tool 装饰器——将函数注册为 LangChain 可调用工具
# ============================================================
@tool(args_schema=CalculatorInput)
def calculator(expression: str, precision: int = 6) -> str:
    """安全地计算数学表达式。支持四则运算、幂运算、三角函数、对数等。
    
    何时调用：
    - 用户要求进行数学计算（如“1234×5678 等于多少”）
    - 用户需要进行科学计算（如“sin(π/2)”）
    - 用户需要单位换算或复杂公式求值
    
    使用指南：
    - 表达式中使用 ** 表示幂运算，如 2**10
    - 三角函数使用弧度制，如 sin(pi/2)
    - 对数使用 log(100, 10) 表示以 10 为底
    
    Args:
        expression: 数学表达式字符串。
        precision: 结果的小数位数（1-12，默认 6 位）。
    """
    # ---- 第一层：正则白名单校验 ----
    valid, err_msg = _validate_expression(expression)
    if not valid:
        return f"⚠️ 输入校验失败：{err_msg}"

    # ---- 第二层：执行与异常捕获 ----
    safe_dict = _build_safe_namespace()

    try:
        # 将 ^ 替换为 **（幂运算兼容）
        cleaned = expression.replace('^', '**')
        result = eval(cleaned, {"__builtins__": {}}, safe_dict)
        rounded = round(result, precision)
        return f"✅ 计算结果：{expression} = {rounded}"
    
    except ZeroDivisionError:
        return "❌ 数学错误：除数不能为零。请检查表达式中的除法运算。"
    except OverflowError:
        return "❌ 计算溢出：结果超出 Python 浮点数范围。请尝试拆分计算或使用对数。"
    except ValueError as e:
        return f"❌ 数值错误：{e}。请检查传入的参数是否正确（例如 sqrt 不能用于负数）。"
    except TypeError as e:
        return f"❌ 类型错误：{e}。请确保所有操作数的类型正确。"
    except SyntaxError as e:
        return f"❌ 语法错误：表达式格式不正确。请检查括号、运算符是否匹配。错误详情：{e}"
    except KeyError as e:
        return f"❌ 函数名错误：'{e}' 不是支持的函数。可用函数：sqrt, sin, cos, tan, log, log2, log10, pi, e, abs, ceil, floor"
    except Exception as e:
        return f"❌ 未知计算错误：{type(e).__name__} —— {e}"


# ============================================================
# 步骤5：独立单元测试（右键 Run 直接验证）
# ============================================================
if __name__ == "__main__":
    test_cases = [
        ("3*4+5",                    "四则运算"),
        ("sqrt(16)*2",               "平方根"),
        ("sin(pi/2)",                "三角函数"),
        ("2**10",                    "幂运算"),
        ("10/0",                     "除零错误"),       # ← 预期友好错误
        ("invalid_expression_!",     "非法字符"),       # ← 预期友好错误
        ("__import__('os').system('ls')", "安全注入"),  # ← 预期被拦截
        ("log(100, 10)",             "对数运算"),
        ("ceil(3.14)",               "取整函数"),
    ]
    print("=" * 60)
    for expr, desc in test_cases:
        print(f"\n【{desc}】输入: {expr}")
        result = calculator.invoke({"expression": expr})
        print(f"  输出: {result}")