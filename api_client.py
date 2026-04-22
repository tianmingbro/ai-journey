# api_client.py (完整版)
import os
import logging
import asyncio
import aiohttp
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------- 抽象基类 -----------------
class BaseAPIClient:
    """定义同步和异步客户端共有的接口行为"""
    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def _build_url(self, endpoint: str) -> str:
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return endpoint

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

# ----------------- 同步客户端 (requests) -----------------
class APIClient(BaseAPIClient):
    """同步 API 客户端，带自动重试"""
    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30):
        super().__init__(api_key, base_url, timeout)
        self.session = requests.Session()
        self.session.headers.update(self._get_headers())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = self._build_url(endpoint)
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        logger.info(f"[SYNC] {method} {url}")
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def post(self, endpoint: str, json_data: dict = None, **kwargs) -> dict:
        try:
            resp = self._request("POST", endpoint, json=json_data, **kwargs)
            return resp.json()
        except Exception as e:
            logger.error(f"POST {endpoint} failed: {e}")
            raise

    def get(self, endpoint: str, params: dict = None, **kwargs) -> dict:
        try:
            resp = self._request("GET", endpoint, params=params, **kwargs)
            return resp.json()
        except Exception as e:
            logger.error(f"GET {endpoint} failed: {e}")
            raise

    def close(self):
        self.session.close()

    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()

# ----------------- 异步客户端 (aiohttp) -----------------
class AsyncAPIClient(BaseAPIClient):
    """异步 API 客户端，支持并发请求"""
    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30):
        super().__init__(api_key, base_url, timeout)
        self._session: aiohttp.ClientSession = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """惰性创建 session，避免在事件循环未启动时创建"""
        if self._session is None or self._session.closed:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=timeout_config
            )
        return self._session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """异步请求核心，支持手动重试（因为 tenacity 对异步装饰器需要特殊处理，这里用循环实现简单重试）"""
        url = self._build_url(endpoint)
        session = await self._get_session()
        # 简单重试循环（异步兼容）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"[ASYNC] {method} {url} (attempt {attempt+1})")
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
                await asyncio.sleep(wait_time)

    async def post(self, endpoint: str, json_data: dict = None, **kwargs) -> dict:
        return await self._request("POST", endpoint, json=json_data, **kwargs)

    async def get(self, endpoint: str, params: dict = None, **kwargs) -> dict:
        return await self._request("GET", endpoint, params=params, **kwargs)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        await self.close()

# ----------------- 模型特定客户端 -----------------

class QwenClient(APIClient):
    """通义千问同步客户端"""
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            timeout=60
        )

    def chat(self, messages: list, model: str = "qwen-turbo", **kwargs) -> str:
        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": kwargs
        }
        resp = self.post("", json_data=payload)
        if resp.get("output") and resp["output"].get("text"):
            return resp["output"]["text"]
        else:
            raise Exception(f"Qwen API error: {resp.get('message', 'Unknown')}")

class ZhipuClient(APIClient):
    """智谱AI同步客户端"""
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            timeout=60
        )

    def chat(self, messages: list, model: str = "glm-4-flash", **kwargs) -> str:
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        resp = self.post("", json_data=payload)
        if resp.get("choices"):
            return resp["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Zhipu API error: {resp.get('error', {}).get('message', 'Unknown')}")

# 异步版本
class AsyncQwenClient(AsyncAPIClient):
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            timeout=60
        )

    async def chat(self, messages: list, model: str = "qwen-turbo", **kwargs) -> str:
        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": kwargs
        }
        resp = await self.post("", json_data=payload)
        if resp.get("output") and resp["output"].get("text"):
            return resp["output"]["text"]
        else:
            raise Exception(f"Async Qwen error: {resp.get('message', 'Unknown')}")

class AsyncZhipuClient(AsyncAPIClient):
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            timeout=60
        )

    async def chat(self, messages: list, model: str = "glm-4-flash", **kwargs) -> str:
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        resp = await self.post("", json_data=payload)
        if resp.get("choices"):
            return resp["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Async Zhipu error: {resp.get('error', {}).get('message', 'Unknown')}")