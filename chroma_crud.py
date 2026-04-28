import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 连接已有向量存储
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            check_embedding_ctx_length=False)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="knowledge_base",
)

# ========== 操作 1：追加新文档 (CREATE) ==========
print("=" * 60)
print("➕ 追加新文档...")
new_docs = [
    Document(
        page_content="RAG（检索增强生成）将信息检索与文本生成结合，显著提升了大模型回答的事实准确性。",
        metadata={"source": "rag_intro", "topic": "AI"},
        id="doc_006"
    ),
    Document(
        page_content="Chroma 是一个开源的轻量级向量数据库，支持本地持久化和客户端-服务端两种部署模式。",
        metadata={"source": "chroma_intro", "topic": "database"},
        id="doc_007"
    ),
]
new_ids = vectorstore.add_documents(new_docs)
print(f"   新增文档 IDs: {new_ids}")

# 验证追加效果
results = vectorstore.similarity_search("RAG", k=1)
if results:
    print(f"   检索验证: {results[0].page_content[:50]}...")
else:
    print("   未找到相关文档")

# ========== 操作 2：删除指定文档 (DELETE) ==========
print("\n" + "=" * 60)
print("🗑️  删除 doc_007 (Chroma 介绍)...")
vectorstore.delete(ids=["doc_007"])
print("   已删除 doc_007")

# 验证删除效果
results_after_delete = vectorstore.similarity_search("Chroma 向量数据库", k=1)
if results_after_delete:
    print(f"   检索验证: {results_after_delete[0].page_content[:50]}...")
else:
    print("   ✅ doc_007 已成功删除，检索无结果")

# ========== 操作 3：更新文档 (UPDATE) ==========
# Chroma 不支持直接更新 → 等价操作：先删除旧文档，再添加新文档
print("\n" + "=" * 60)
print("✏️ 更新 doc_002 (天气报告)...")

# 第 1 步：删除旧文档
vectorstore.delete(ids=["doc_002"])

# 第 2 步：用相同 ID 添加新文档（内容和元数据均更新）
updated_doc = Document(
    page_content="今天北京天气多云转晴，最高气温 18 摄氏度，下午有微风，适合户外活动。",
    metadata={"source": "weather_report_v2", "topic": "weather", "version": "2.0"},
    id="doc_002"
)
vectorstore.add_documents([updated_doc])

# 验证更新效果
updated_result = vectorstore.similarity_search("今天天气", k=1)
if updated_result:
    doc = updated_result[0]
    print(f"   更新后内容: {doc.page_content[:80]}...")
    print(f"   元数据: {doc.metadata}")
    # 确认版本字段已更新
    assert doc.metadata.get("version") == "2.0", "版本号未更新！"
    print("   ✅ 内容与元数据均已更新")

# ========== 操作 4：查看最终统计 ==========
print("\n" + "=" * 60)
print("📋 集合最终统计")
print(f"   文档总数: {vectorstore._collection.count()}")

# 打印所有文档概况
all_data = vectorstore.get()
print("   文档列表:")
for i, doc_id in enumerate(all_data["ids"]):
    content = all_data["documents"][i][:40]
    meta = all_data["metadatas"][i]
    print(f"     [{doc_id}] {content}... (topic: {meta.get('topic')})")

print("\n✅ 增、删、改三项操作全部完成！")