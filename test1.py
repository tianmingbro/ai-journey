from langchain_chroma import Chroma
from utils.embedding import get_embeddings
from config.settings import settings

# 加载您当前的向量库
vectorstore = Chroma(
    embedding_function=get_embeddings(),
    persist_directory=settings.chroma_persist_dir,
    collection_name=settings.collection_name
)
# 获取所有文档，看看有没有电商相关的内容
all_data = vectorstore.get()
print(f"向量库中总共有 {len(all_data['ids'])} 个文本块")
for i, doc in enumerate(all_data['documents']):
    if '电商' in doc or '优惠券' in doc or '购物车' in doc:
        print(f"[!] 找到相关片段: {doc[:100]}...")
if not any('电商' in str(doc) for doc in all_data['documents']):
    print("⚠️ 警告：向量库中完全没有找到电商相关的文档！")