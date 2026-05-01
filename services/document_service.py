from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_core.documents import Document
from utils.text_splitter import get_splitter

class DocumentService:
    """负责从本地目录加载文档并分割为文本块"""

    @staticmethod
    def load_and_split(docs_dir:str)->List[Document]:
        #1.加载所有.txt/.md文件
        loader=DirectoryLoader(
            path=docs_dir,
            glob=["**/*.txt","**/*.md"],
            loader_cls=lambda fp:TextLoader(fp,encoding='utf-8'),
            show_progress=True,
            use_multithreading=True
        )
        raw_docs=loader.load()
        #2.分割文档
        splitter=get_splitter()
        chunks=splitter.split_documents(raw_docs)

        #3.为每个块补充文件名等元数据
        for chunk in chunks:
            source=chunk.metadata.get("source","")
            if source:
                chunk.metadata["filename"]=Path(source).name
            chunk.metadata["processed_by"]="DocumentService"
        
        return chunks
    
    # services/document_service.py (新增方法)

class DocumentService:
    # ... 原有的 load_and_split 方法保持不变 ...

    @staticmethod
    def load_file_and_split(file_path: str) -> List[Document]:
        """
        加载单个文本文档，并将其分割为文本块。
        Args:
            file_path: 单个文件的绝对路径（必须是 .txt/.md 等纯文本格式）
        Returns:
            分割后的 Document 列表
        """
        # 使用 TextLoader 直接加载单个文件，避免 DirectoryLoader 的目录扫描逻辑
        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()

        # 使用相同的分割器进行分块
        splitter = get_splitter()
        chunks = splitter.split_documents(raw_docs)

        # 补充元数据
        for chunk in chunks:
            chunk.metadata['filename'] = Path(file_path).name
            chunk.metadata['processed_by'] = 'DocumentService'

        return chunks