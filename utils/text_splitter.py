from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

def get_splitter():
    """返回配置好的RecursiveCharacterTextSplitter实例"""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n","\n","。",".","！","!","？","?"," ",""]
    )