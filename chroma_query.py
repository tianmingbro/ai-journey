import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List,Tuple,Optional,Dict,Any

#封装chroma管理器
class ChromaManager:
    """封装chroma的常用查询和管理操作"""
    def __init__(
            self,
            persist_directory:str="./chroma_db",
            collection_name:str="knowledge_base",
            embedding_model:str="text-embedding-v3"
            ):
        self.embeddings=OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            check_embedding_ctx_length=False)
        
        self.vectorstore=Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )    
        #基础相似度搜索
    def query(self,text:str,k:int=3,filter:Optional[Dict[str,Any]]=None) ->List[Document]:
        """基础语义搜索，返回最相似的k个文档"""
        return self.vectorstore.similarity_search(text,k=k,filter=filter)
    
    #带分数的相似度搜索
    def query_with_scores(self,text:str,k:int=3,filter:Optional[Dict[str,Any]]=None) ->List[Tuple[Document,float]]:
        """返回（Document,distance_score)的列表，分数越小越相关"""
        return self.vectorstore.similarity_search_with_score(text,k=k,filter=filter)
    #获取全部文档（用于调试或审查）
    def get_all(self,limit:int=100)->List[Document]:
        """获取集合中所有文档"""
        #通过collection直接获取，返回dict需转换
        data=self.vectorstore.get(limit=limit)
        docs=[]
        for i in range(len(data["ids"])):
            docs.append(Document(
                page_content=data["documents"][i],
                metadata=data["metadatas"][i] if data["metadatas"] else{},
                id=data["ids"][i]
            ))
        return docs
    #使用as_retiever转换为标准检索器
    def as_retriever(self,k:int=3,filter:Optional[Dict[str,Any]]=None):
        """返回langchain retriever对象，可直接入链"""
        search_kwargs={"k":k}
        if filter:
            search_kwargs["filter"]=filter
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    def count(self)->int:
        return self.vectorstore._collection.count()
#主程序：三种查询方式演示
if __name__=="__main__":
    #1.初始化管理器
    manager=ChromaManager(
        persist_directory="./chroma_db",
        collection_name="knowledge_base"
    )
    print(f"集合文档总数：{manager.count()}\n")
    #查询方式1：基础相似度搜索
    print("基础相似度搜索")
    results=manager.query("模型参数",k=3)
    for i,doc in enumerate(results,1):
        print(f"    #{i}|主题：{doc.metadata.get('topic')}|来源：{doc.metadata.get('source')}")
        print(f"    内容：{doc.page_content[:80]}...\n")
    # ========== 查询方式 2：带分数的相似度搜索 ==========
    print("=" * 60)
    print("📊 带分数搜索: '今天天气'")
    print("-" * 60)
    scored_results = manager.query_with_scores("今天天气", k=3)
    for i, (doc, score) in enumerate(scored_results, 1):
        print(f"  #{i} | 距离: {score:.4f} | 主题: {doc.metadata.get('topic')}")
        print(f"     内容: {doc.page_content[:80]}...\n")

    # ========== 查询方式 3：元数据过滤 ==========
    print("=" * 60)
    print("🔎 元数据过滤搜索: '机器学习' (仅 topic='AI')")
    print("-" * 60)
    filtered = manager.query("机器学习", k=3, filter={"topic": "AI"})
    for i, doc in enumerate(filtered, 1):
        print(f"  #{i} | 主题: {doc.metadata['topic']} | 来源: {doc.metadata['source']}")
        print(f"     内容: {doc.page_content[:80]}...\n")

    #转换为检索器演示
    print("使用as_retriever（）转换为标准检索器（k=2）")
    retriever=manager.as_retriever(k=2)
    retrieved_docs=retriever.invoke("人工智能")
    for i ,doc in enumerate(retrieved_docs,1):
        print(f"    #{i}|{doc.page_content[:80]}...")
    
    # ========== 获取全部文档 ==========
    print("\n" + "=" * 60)
    print("📋 获取全部文档 (get_all)")
    print("-" * 60)
    all_docs = manager.get_all()
    for doc in all_docs:
        print(f"  [{doc.id}] {doc.page_content[:60]}...")

    print("\n✅ 三种查询方式验证完成！")