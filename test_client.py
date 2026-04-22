# test_clients.py
import asyncio
from api_client import ZhipuClient, AsyncZhipuClient, QwenClient, AsyncQwenClient

def test_sync():
    print("=== 同步智谱调用 ===")
    with ZhipuClient() as client:
        reply = client.chat([{"role": "user", "content": "用一句话介绍异步编程。"}])
        print("智谱:", reply)

    print("=== 同步通义调用 ===")
    with QwenClient() as client:
        reply = client.chat([{"role": "user", "content": "用一句话介绍异步编程。"}])
        print("通义:", reply)

async def test_async():
    print("=== 异步智谱调用 ===")
    async with AsyncZhipuClient() as client:
        reply = await client.chat([{"role": "user", "content": "用一句话介绍异步编程。"}])
        print("智谱异步:", reply)

    print("=== 异步通义调用 ===")
    async with AsyncQwenClient() as client:
        reply = await client.chat([{"role": "user", "content": "用一句话介绍异步编程。"}])
        print("通义异步:", reply)

async def test_concurrent():
    """演示并发调用多个模型"""
    async def call_zhipu():
        async with AsyncZhipuClient() as client:
            return await client.chat([{"role": "user", "content": "你好"}])
    async def call_qwen():
        async with AsyncQwenClient() as client:
            return await client.chat([{"role": "user", "content": "你好"}])

    results = await asyncio.gather(call_zhipu(), call_qwen())
    print("智谱:", results[0])
    print("通义:", results[1])

if __name__ == "__main__":
    test_sync()
    asyncio.run(test_async())
    asyncio.run(test_concurrent())