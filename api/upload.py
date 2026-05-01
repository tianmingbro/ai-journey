import os
import re
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from api.models import UploadResponse
from services.document_service import DocumentService
from utils.text_splitter import get_splitter          # 复用工具层的分割器
from services.vector_store_service import VectorStoreService

router=APIRouter()

#只允许纯文本与markdown，防止任意文件写入
ALLOWED_EXTENSIONS={".txt",".md",".markdown"
                    }
@router.post("/upload",response_model=UploadResponse)
async def upload_document(file: UploadFile=File(...)):
    #1.文件类型校验
    _,ext=os.path.splitext(file.filename or "")
    ext=ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型’{ext}'。仅允许：{'，'.join(ALLOWED_EXTENSIONS)}"
        )
    #2.安全文件名
    #从content-disposition获取的原始文件名可能包含路径信息，做一次清洗
    raw_name=file.filename or "upload"
    safe_name=re.sub(r'[\\/*?:"<>|]',"_",raw_name)

    #3.读取内容并写入临时文件
    try:
        content_bytes=await file.read()
        conten_str=content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400,detail="文件编码不是UTF-8，请转换后上传。")
    if not content_bytes.strip():
        raise HTTPException(status_code=400, detail="文件内容为空。")
    #使用临时文件让documentservice处理（他通过文件路径加载）
    tmp_path=None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=ext,
            delete=False,
            encoding="utf-8"
        ) as tmp:
            tmp.write(conten_str)
            tmp_path=tmp.name
        # # 使用 TextLoader 加载单个文件
        # loader = TextLoader(tmp_path, encoding="utf-8")
        # docs = loader.load()    
       
        #  # 分割文档
        # splitter = get_splitter()
        # chunks = splitter.split_documents(docs)
        # if not chunks:
        #     raise HTTPException(status_code=400, detail="无法从文件中提取有效文本块。")
         # ---------- 4. 加载文档并分块 ----------
        docs = DocumentService.load_file_and_split(tmp_path)  
        #5.写入向量数据库
        VectorStoreService.create_from_documents(docs)
        return UploadResponse(
            filename=safe_name,
            chunks=len(docs),
            status="success",
            detail=""
        )
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"处理文件时出错：{str(e)}")
    finally:
        # 无论如何都要清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)