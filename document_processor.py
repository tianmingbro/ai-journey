from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_documents(
    docs_dir: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    file_glob: str = ["**/*.txt", "**/*.md"],
    separators: Optional[List[str]] = None,
    encoding: str = "utf-8"
) -> List[Document]:
    """加载目录中的所有文本文档，并分割为语义块。

    Args:
        docs_dir: 文档目录路径。
        chunk_size: 每个文本块的最大字符数。
        chunk_overlap: 相邻块的重叠字符数。
        file_glob: 文件名匹配模式。
        separators: 递归分割分隔符列表，默认为中英文混合。
        encoding: 文本文件编码。

    Returns:
        List[Document]: 分割后的文档块列表，每个块携带元数据。
    """
    # ---------- 阶段1：加载 ----------
    print(f"📂 正在加载目录: {docs_dir}")
    loader = DirectoryLoader(
        path=docs_dir,
        glob=file_glob,
        loader_cls=lambda fp: TextLoader(fp, encoding=encoding),
        show_progress=True,
        use_multithreading=True
    )
    raw_docs = loader.load()
    print(f"   加载完成，共 {len(raw_docs)} 个原始文档。")

    if not raw_docs:
        print("⚠️  未发现任何匹配文档，返回空列表。")
        return []

    # ---------- 阶段2：分割 ----------
    if separators is None:
        separators = [
            "\n\n",   # 段落
            "\n",     # 换行
            "。",     # 中文句号
            ".",      # 英文句号
            "！", "!",
            "？", "?",
            " ",      # 空格
            ""        # 字符级兜底
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=separators
    )

    chunks = splitter.split_documents(raw_docs)
    print(f"   分割完成，共 {len(chunks)} 个文本块。")

    # ---------- 阶段3：添加自定义元数据 ----------
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        # 提取简短文件名
        filename = Path(source).name
        # 提取文件后缀
        suffix = Path(source).suffix.lower()
        # 记录处理信息
        chunk.metadata.update({
            "filename": filename,
            "file_type": suffix,
            "chunk_size": len(chunk.page_content),
            "processed_by": "document_processor_v1"
        })

    print(f"   已为所有块添加自定义元数据 (filename, file_type, chunk_size)。")
    return chunks

# ==================== 测试 ====================
if __name__ == "__main__":
    # 假设文档放在 ./test_docs 下
    import os
    docs_path = "./test_docs"
    if not os.path.isdir(docs_path):
        print(f"❌ 目录不存在: {docs_path}")
        print("   请创建该目录并放入 .txt 或 .md 文件。")
    else:
        result_chunks = process_documents(docs_path)

        print("\n📋 结果预览 (前3个块):")
        for i, chunk in enumerate(result_chunks[:3]):
            print(f"--- Chunk {i+1} ---")
            print(f"  来源文件: {chunk.metadata['filename']}")
            print(f"  文件类型: {chunk.metadata['file_type']}")
            print(f"  块大小: {chunk.metadata['chunk_size']} 字符")
            print(f"  起始位置: {chunk.metadata['start_index']}")
            print(f"  内容: {chunk.page_content[:80]}...")
            print()