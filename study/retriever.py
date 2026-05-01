import numpy as np
from langchain_openai import OpenAIEmbeddings
from typing import List,Tuple

class NumpyRetriever:
    """基于Numpy的极简向量检索器，模拟向量数据库的核心流程"""

    def __init__(self,embedding_model:OpenAIEmbeddings):
        self.embeddings=embedding_model
        self.documents:List[str]=[]
        self.vectors:np.ndarray=None

    def add_documents(self,docs:List[str]) -> None:
        """将文档列表嵌入并追加到索引"""
        if not docs:
            return
        new_vecs=self.embeddings.embed_documents(docs)
        new_mat=np.array(new_vecs)
        if self.vectors is None:
            self.vectors=new_mat
        else:
            self.vectors=np.vstack([self.vectors,new_mat])
        self.documents.extend(docs)
    def similarity_search(self,query:str,k: int=3) ->List[Tuple[str,float]]:
        """返回与查询最相似的k个文档片段及相似度"""
        if self.vectors is None or len(self.documents)==0:
            return[]
        #查询向量并归一化
        q_vec=np.array(self.embeddings.embed_query(query))
        q_norm=q_vec/np.linalg.norm(q_vec)
        #文档矩阵归一化（每行是一个文档向量）
        doc_norms=np.linalg.norm(self.vectors,axis=1,keepdims=True)
        d_norm=self.vectors/doc_norms
        #余弦相似度=归一化矩阵点积
        scores=(d_norm@q_norm).flatten()
        #取top k
        top_k_indices=np.argsort(-scores)[:k]
        return [(self.documents[i],float(scores[i])) for i in top_k_indices]
    def __len__(self):
        return len(self.documents)

if __name__=="__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    #初始化Embedding模型
    emb=OpenAIEmbeddings(
        model="text-embedding-v3",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False
    )
    #创建检索器
    retriever=NumpyRetriever(emb)
    #模拟知识库：一组不同主题的文档片段
    docs=["大语言模型通常包含数十亿甚至数千亿参数，需要海量数据和计算资源进行训练。",
        "今天北京天气晴朗，最高气温 22 摄氏度，适合户外运动和郊游。",
        "向量数据库是一种专门用于存储和检索高维向量的数据库系统，常用于语义搜索和推荐系统。",
        "Transformer 架构中的自注意力机制是 LLM 的核心组件，由 Vaswani 等人在 2017 年提出。",
        "Python 是数据科学和机器学习领域最流行的编程语言之一，拥有丰富的生态库。",
        "苹果公司今天发布了新款 iPhone，引发了全球科技爱好者的关注。",]
    retriever.add_documents(docs)
    print(f"一索引{len(retriever)}个文档片段\n")

    #测试1：模型参数相关
    print("查询：‘模型参数’")
    for doc,score in retriever.similarity_search("模型参数",k=3):
        bar=" "*max(0,int(score*50))
        print(f"  [{score:.4f}]{doc}  {bar}")  

    # 测试 2：天气相关
    print("\n🔍 查询: '今天天气如何'")
    for doc, score in retriever.similarity_search("今天天气如何", k=3):
        bar = "█" * max(0, int(score * 50))
        print(f"  [{score:.4f}] {doc}  {bar}")

    # 测试 3：另一个主题
    print("\n🔍 查询: 'iPhone 新品'")
    for doc, score in retriever.similarity_search("iPhone 新品", k=3):
        bar = "█" * max(0, int(score * 50))
        print(f"  [{score:.4f}] {doc}  {bar}")                  
