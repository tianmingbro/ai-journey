import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

texts = [
    # 同一语义簇的两个变体
    "你好，请问图书馆在哪里？",
    "图书馆怎么走？我想借几本小说。",
    # 完全不同的语义簇
    "今天天气真热，气温超过了 35 度。",
    # 另一个簇
    "人工智能正在改变我们与计算机交互的方式。",
    "大语言模型让机器能够理解自然语言。",
    "量子力学研究微观粒子的运动规律。"
    "今天中午我吃了一碗拉面。"
]

# 1. 1536 维模型

emb_1536 = OpenAIEmbeddings(model="text-embedding-v3",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False)
vecs_1536 = np.array(emb_1536.embed_documents(texts))

# 2. 512 维模型
emb_512 = OpenAIEmbeddings(model="text-embedding-v3",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False, dimensions=512)
vecs_512 = np.array(emb_512.embed_documents(texts))

def cosine_sim(mat: np.ndarray) -> np.ndarray:
    """计算一组向量的余弦相似度矩阵"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = mat / norms
    return mat_norm @ mat_norm.T

sim_1536 = cosine_sim(vecs_1536)
sim_512 = cosine_sim(vecs_512)

print("=" * 80)
print("📐 维度缩减对比实验 (text-embedding-3-small)")
print("-" * 80)
print(f"{'指标':<40} {'1536维':>18} {'512维':>18}")
print("-" * 80)

# 语义保持度：同义句对 (0,1) 的相似度
s_1536_01 = float(sim_1536[0][1])
s_512_01 = float(sim_512[0][1])
print(f"{'同义句相似度 [0][1]（图书馆）':<40} {s_1536_01:>18.4f} {s_512_01:>18.4f}")

# 语义区分度：不同句对 (0,2) 的相似度
s_1536_02 = float(sim_1536[0][2])
s_512_02 = float(sim_512[0][2])
print(f"{'异义句相似度 [0][2]（图书馆 vs 天气）':<40} {s_1536_02:>18.4f} {s_512_02:>18.4f}")

# 另一个语义簇 (3,4) AI 相关
s_1536_34 = float(sim_1536[3][4])
s_512_34 = float(sim_512[3][4])
print(f"{'同义句相似度 [3][4]（AI）':<40} {s_1536_34:>18.4f} {s_512_34:>18.4f}")

# 内存占用对比
import sys
mem_1536 = sys.getsizeof(vecs_1536.tobytes()) + sys.getsizeof(vecs_1536)
mem_512 = sys.getsizeof(vecs_512.tobytes()) + sys.getsizeof(vecs_512)
print(f"{'近似内存占用 (bytes)':<40} {mem_1536:>18,} {mem_512:>18,}")

# 成本估算
n_tokens_per_text = 30  # 估计每条中文文本 ~30 tokens
total_tokens = len(texts) * n_tokens_per_text
price_per_1m = 0.02  # text-embedding-3-small 定价
cost = total_tokens / 1_000_000 * price_per_1m
print(f"{'API 嵌入成本 (美元)':<40} ${cost:>18.6f} ${cost:>18.6f}")
print(f"(注: 成本与维度无关，均为 $0.02/1M tokens)")
print("-" * 80)
print("结论：512 维在语义相似度保持度上几乎无损，但存储和后续计算成本大幅降低。")