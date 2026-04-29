# chunking_strategies.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Callable, Dict

def strategy_conservative(docs: List[Document]) -> List[Document]:
    """保守策略：小块 (256)，重叠 10%"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=26,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        # 使用父文档 ID + 序号
        parent_id = chunk.metadata.get("id", "doc_000")
        chunk.id = f"{parent_id}_chunk_{i:03d}"
    return chunks

def strategy_standard(docs: List[Document]) -> List[Document]:
    """标准策略：中块 (512)，重叠 15%"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=77,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        # 使用父文档 ID + 序号
        parent_id = chunk.metadata.get("id", "doc_000")
        chunk.id = f"{parent_id}_chunk_{i:03d}"
    return chunks

def strategy_aggressive(docs: List[Document]) -> List[Document]:
    """大块策略：大块 (1024)，重叠 20%"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=205,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        # 使用父文档 ID + 序号
        parent_id = chunk.metadata.get("id", "doc_000")
        chunk.id = f"{parent_id}_chunk_{i:03d}"
    return chunks

# 统一策略字典，方便循环调用
STRATEGIES: Dict[str, Callable] = {
    "保守策略 (256/10%)": strategy_conservative,
    "标准策略 (512/15%)": strategy_standard,
    "大块策略 (1024/20%)": strategy_aggressive,
}