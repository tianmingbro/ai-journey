#!/usr/bin/env python3
"""
api_client.py - 使用 requests 库封装大模型 API 调用
支持平台：通义千问（DashScope）等
"""

import os
import json
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class LLMClient:
    """通用大模型客户端（基于 requests 实现）"""
    
    def __init__(self, api_key=None, base_url=None, model=None):
        """
        初始化客户端
        :param api_key: API Key，默认从环境变量 DASHSCOPE_API_KEY 读取
        :param base_url: API 基础 URL，默认使用通义千问 DashScope 地址
        :param model: 模型名称，默认 qwen-plus
        """
        self.api_key = api_key or os.getenv('ZHIPU_API_KEY')
        if not self.api_key:
            raise ValueError("请在 .env 文件中设置 ZHIPU_API_KEY")
        
        self.base_url = base_url or "https://open.bigmodel.cn/api/paas/v4/"
        self.model = model or os.getenv('MODEL_NAME', 'glm-4-flash')
        
        # 请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages, temperature=0.7, max_tokens=1000):
        """
        发送对话请求
        :param messages: 消息列表，格式 [{"role": "user", "content": "你好"}]
        :param temperature: 温度参数 (0~1)
        :param max_tokens: 最大输出 token 数
        :return: 模型返回的文本内容
        """
        # 构造请求体
        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": "message"   # 返回完整消息格式
            }
        }
        
        try:
            # 发送 POST 请求
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # 检查 HTTP 状态码
            response.raise_for_status()
            
            # 解析 JSON 响应
            result = response.json()
            
            # 提取回答内容
            # 响应结构: output.choices[0].message.content
            if 'output' in result and 'choices' in result['output']:
                answer = result['output']['choices'][0]['message']['content']
                return answer
            else:
                # 如果格式不符，打印完整响应用于调试
                error_msg = f"响应格式异常: {json.dumps(result, ensure_ascii=False)}"
                raise RuntimeError(error_msg)
                
        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时，请稍后重试")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("网络连接失败，请检查网络")
        except requests.exceptions.HTTPError as e:
            status_code = response.status_code
            if status_code == 401:
                raise RuntimeError("API Key 无效，请检查 .env 文件")
            elif status_code == 429:
                raise RuntimeError("请求频率超限，请稍后再试")
            else:
                raise RuntimeError(f"HTTP 错误 {status_code}: {e}")
        except json.JSONDecodeError:
            raise RuntimeError("响应不是有效的 JSON 格式")
        except Exception as e:
            raise RuntimeError(f"调用失败: {e}")

# ---------- 简单交互式对话示例 ----------
def interactive_chat():
    """交互式对话（保留上下文）"""
    client = LLMClient()
    print("对话助手（输入 exit 退出）")
    messages = []
    
    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() == 'exit':
            print("再见！")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            answer = client.chat(messages)
            print(f"AI：{answer}")
            messages.append({"role": "assistant", "content": answer})
        except RuntimeError as e:
            print(f"错误：{e}")

if __name__ == "__main__":
    # 直接运行本脚本时启动交互式对话
    interactive_chat()