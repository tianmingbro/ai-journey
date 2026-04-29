from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import documents  # 复用文档加载步骤的结果

# 定义分词器使用的分隔符（针对中英文混合优化）
CUSTOM_SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 换行
    "。",     # 中文句号
    ".",      # 英文句号
    "！",     # 中文感叹号
    "!",      # 英文感叹号
    "？",     # 中文问号
    "?",      # 英文问号
    " ",      # 空格
    ""        # 字符级（兜底）
]

print(f"原始文档数: {len(documents)}")
total_chars = sum(len(doc.page_content) for doc in documents)
print(f"总字符数: {total_chars}\n")

# =====================================================
# 配置 A：小块 + 少重叠
# =====================================================
splitter_a = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
    separators=CUSTOM_SEPARATORS
)

chunks_a = splitter_a.split_documents(documents)

print("=" * 60)
print("配置 A：chunk_size=100, chunk_overlap=20")
print(f"生成 {len(chunks_a)} 个文本块\n")

for i, chunk in enumerate(chunks_a[:4], 1):
    print(f"Chunk #{i}")
    print(f"  来源: {chunk.metadata['source']}")
    print(f"  起始索引: {chunk.metadata['start_index']}")
    print(f"  内容: {chunk.page_content}")
    print()

# 展示重叠区域（选取连续两块的边界）
if len(chunks_a) >= 2:
    print("--- 重叠区域观察 (Chunk #1 尾 + Chunk #2 头) ---")
    print(f"Chunk #1 结尾: {chunks_a[0].page_content[-40:]}...")
    print(f"Chunk #2 开头: {chunks_a[1].page_content[:40]}...")
    print()

# =====================================================
# 配置 B：大块 + 多重叠
# =====================================================
splitter_b = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=80,
    length_function=len,
    add_start_index=True,
    separators=CUSTOM_SEPARATORS
)

chunks_b = splitter_b.split_documents(documents)

print("=" * 60)
print("配置 B：chunk_size=200, chunk_overlap=80")
print(f"生成 {len(chunks_b)} 个文本块\n")

for i, chunk in enumerate(chunks_b[:3], 1):
    print(f"Chunk #{i}")
    print(f"  来源: {chunk.metadata['source']}")
    print(f"  起始索引: {chunk.metadata['start_index']}")
    print(f"  内容: {chunk.page_content}")
    print()

if len(chunks_b) >= 2:
    print("--- 重叠区域观察 (Chunk #1 尾 + Chunk #2 头) ---")
    print(f"Chunk #1 结尾: ...{chunks_b[0].page_content[-40:]}")
    print(f"Chunk #2 开头: {chunks_b[1].page_content[:40]}...")
    print()

# =====================================================
# 对比总结
# =====================================================
print("=" * 60)
print("📊 参数影响分析")
print(f"配置 A (100/20): 块数={len(chunks_a)}, 平均块大小≈{total_chars/len(chunks_a):.0f} 字符")
print(f"配置 B (200/80): 块数={len(chunks_b)}, 平均块大小≈{total_chars/len(chunks_b):.0f} 字符")
print()
print("结论：")
print("- chunk_size 越小 → 块数越多，单块语义更聚焦，但可能切断连贯表达。")
print("- chunk_overlap 越大 → 相邻块共享更多上下文，降低信息断裂风险，但增加冗余存储。")
print("- 中文场景需在 separators 中包含中文标点，否则可能在英文标点处强行切分。")
# 接在原有代码之后
print("\n===== 诊断信息 =====")
print(f"文档数量: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"文档{i}长度: {len(doc.page_content)} 字符")

# 强制用极短的分块演示重叠
demo_splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=10,
    separators=["\n", "。", ".", " "]
)
demo_chunks = demo_splitter.split_documents(documents)

if len(demo_chunks) >= 2:
    print(f"演示分块数: {len(demo_chunks)}")
    print("Chunk 0 尾: ", repr(demo_chunks[0].page_content[-30:]))
    print("Chunk 1 头: ", repr(demo_chunks[1].page_content[:30]))
else:
    print("文档过短，无法产生多个块，请用更长的文本测试")