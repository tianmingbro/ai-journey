import os
import numpy as np
from langchain_openai import OpenAIEmbeddings

# ========== 1. 请你在运行前，确保这部分的配置是正确的 ==========  
# 方法一：直接填写你的API Key（用真实Key替换下面的占位符）
API_KEY = "sk-991aa8d5210f42fab50ce7f59dfca11a"

# 方法二：从环境变量读取（推荐） 
# 先在终端运行 export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# 然后取消下面一行的注释，该代码将从环境变量中读取Key[reference:5]：
# API_KEY = os.getenv("DASHSCOPE_API_KEY") 

# 根据你的API Key地域，注释掉错误的那个base_url
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"         # 北京地域
# BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # 新加坡（国际站）

# 根据你的平台，确认模型名称
MODEL_NAME = "text-embedding-v3"         # 适用于阿里云百炼
# MODEL_NAME = "BAAI/bge-m3"             # 适用于硅基流动

# ========== 2. 调试信息与初始化 ==========  
print(f"当前API Key前6位: ...{API_KEY[-6:]}")  # 不要打印完整Key，出于安全考虑
print(f"当前base_url: {BASE_URL}")

try:
    embeddings = OpenAIEmbeddings(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        check_embedding_ctx_length=False
    )
    # 尝试获取向量维度，如果失败会抛出异常
    print(f"Embeddings初始化成功，模型: {MODEL_NAME}")
except Exception as e:
    print(f"Embeddings初始化失败，请检查Key和网络配置: {e}")
    exit()

# ========== 3. 以下是你原有的核心逻辑，保持不变 ==========
def compute_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_similarity_matrix(vectors):
    mat = np.array(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = mat / norms
    return mat_norm @ mat_norm.T



# ========== 测试语料：跨语言问候 + 语义干扰句 ==========
sentences = [
    # 问候簇（中/英/日）
    "你好，很高兴见到你",                # 0
    "Hello, nice to meet you",          # 1
    "こんにちは、お会いできて嬉しいです",  # 2
    "您好，见到您很荣幸",                # 3
    # 天气簇
    "今天天气真不错，阳光明媚",          # 4
    "The weather is beautiful today",   # 5
    # 科技簇
    "人工智能正在改变世界",              # 6
    "Artificial intelligence is transforming the world", # 7
    # 其他
    "我喜歡吃蘋果",                     # 8 （繁体中文）
    "I like to eat apples",            # 9
]

print("=" * 70)
print("📐 向量化结果")
print("-" * 70)
vectors = embeddings.embed_documents(sentences)
print(f"模型维度: {len(vectors[0])}")  # 预期 1536

# 输出每个句子的前5个向量值（便于观察）
for i, (text, vec) in enumerate(zip(sentences, vectors)):
    print(f"[{i}] {text}")
    print(f"    前5个分量: {[round(v, 4) for v in vec[:5]]}")

print("\n" + "=" * 70)
print("📊 余弦相似度矩阵（纯 NumPy）")
print("-" * 70)
sim = compute_similarity_matrix(vectors)

# 打印重点对比
pairs = [
    (0, 1, "中文你好 ↔ 英文Hello"),
    (0, 2, "中文你好 ↔ 日文こんにちは"),
    (0, 3, "中文你好 ↔ 中文您好"),
    (0, 4, "中文你好 ↔ 天气句子(中)"),
    (0, 5, "中文你好 ↔ 天气句子(英)"),
    (6, 7, "中文AI ↔ 英文AI"),
    (8, 9, "繁体我喜欢吃苹果 ↔ 英文I like apples"),
    (4, 5, "中文天气 ↔ 英文天气"),
]
for i, j, desc in pairs:
    s = float(sim[i][j])
    bar = "█" * max(0, int(s * 50))
    print(f"  {desc:<35} 相似度: {s:.4f}  {bar}")

# ========== 单句查询演示 ==========
print("\n" + "=" * 70)
print("🔍 查询：'机器学习是什么？'")
print("-" * 70)
query = "机器学习是什么？"
query_vec = np.array(embeddings.embed_query(query))
doc_mat = np.array(vectors)
# 归一化
q_norm = query_vec / np.linalg.norm(query_vec)
d_norm = doc_mat / np.linalg.norm(doc_mat, axis=1, keepdims=True)
scores = (d_norm @ q_norm).flatten()

# 按相似度从高到低排序
for rank, idx in enumerate(np.argsort(-scores)):
    print(f"  #{rank+1} [{float(scores[idx]):.4f}] {sentences[idx]}")