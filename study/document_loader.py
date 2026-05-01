import os
from langchain_community.document_loaders import DirectoryLoader,TextLoader

#1.定义文档目录路径
DOCS_DIR="./test_docs"
#2.自定义一个textloader工厂函数
def create_text_loader(file_path,**kwargs):
    return TextLoader(file_path,encoding="utf-8")

#3.配置directoryloader
#    - path: 要扫描的目录
#    - glob: 匹配文件的规则，这里加载所有 .txt 和 .md 文件
#    - loader_cls: 告诉 DirectoryLoader 对每个匹配到的文件使用哪个加载器
#    - show_progress: 显示进度条（文件多时可以看到进度）
#    - use_multithreading: 启用多线程加速加载
loader=DirectoryLoader(
    path=DOCS_DIR,
    glob=["**/*.txt", "**/*.md"],
    loader_cls=create_text_loader,
    show_progress=True,
    use_multithreading=True
)
#4.执行加载
print(f"正在从’{DOCS_DIR}加载文档。。")
documents=loader.load()
print(f"加载完成，共找到{len(documents)}篇文档。\n")
#5.打印每一篇文档的元数据和内容预览
for i,doc in enumerate(documents,1):
    print(f"---文档{i}---")
    print(f"文档路径：{doc.metadata['source']}")
    print(f"内容预览：{doc.page_content[:100]}...")