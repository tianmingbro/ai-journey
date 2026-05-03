# 🤖 RAG 知识库问答系统

基于 **LangChain + Chroma + FastAPI + Streamlit** 的检索增强生成（RAG）问答系统，支持多轮对话、流式输出、文档上传、评测埋点与容器化部署。

<!-- 自动显示 GitHub Stars 数量 -->
![GitHub stars](https://img.shields.io/github/stars/你的用户名/仓库名?style=social)

<!-- 自动显示 PyPI 最新版本 -->
![PyPI version](https://img.shields.io/pypi/v/你的包名)

<!-- 自动显示 Docker 镜像大小 -->
![Docker image size](https://img.shields.io/docker/image-size/你的用户名/镜像名)

---

## 📖 简介

本项目实现了一套完整的 **RAG (Retrieval-Augmented Generation)** 问答管线：

1. 用户上传 `.txt` / `.md` 文档，系统自动分块、向量化并存入 Chroma 向量数据库。  
2. 用户提问时，系统检索语义最相关的文档片段，拼接上下文后由大模型生成精准回答。  
3. 支持**流式输出（SSE）**，提供类似 ChatGPT 的逐字显示体验。  
4. 内置**多轮对话记忆**，用户可连续追问。  
5. 集成**评测埋点系统**，自动记录每次问答的检索文档、延迟、Token 消耗等数据，为批量评测提供数据支撑。  
6. 使用 **Docker Compose + Nginx** 实现一键部署，统一域名入口，支持 WebSocket 代理。

---

## ✨ 主要特性

- **📁 文档管理**：上传任意 `.txt` / `.md` 文件，自动分割并写入向量数据库。
- **🔍 语义检索**：基于 `text-embedding-v3` 的稠密向量检索，支持余弦相似度匹配。
- **🧠 上下文感知**：LLM 结合检索到的文档片段与对话历史生成答案，有效抑制幻觉。
- **💬 多轮对话**：前端自动维护会话 ID，后端通过 `ChatMessageHistory` 管理对话上下文。
- **⚡ 流式输出**：Server-Sent Events (SSE) 实时推送回答 token，提升交互体验。
- **📊 评测埋点**：每次问答自动记录检索结果、耗时、Token 消耗等，支持 Hit Rate、MRR、Faithfulness 离线计算。
- **🐳 容器化部署**：提供完整 Dockerfile 及 docker-compose.yml，搭配 Nginx 反向代理，开箱即用。

---

## 🧱 系统架构
Browser (http://your-domain.com)
│
▼
┌──────────┐
│ Nginx │ (反向代理, 80/443)
└────┬─────┘
│ /app ────────────┐
│ /api ────┐ │
▼ ▼ ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Streamlit│ │ FastAPI │ │ Chroma (卷) │
│ :8501 │ │ :8000 │ │ /app/chroma_ │
└──────────┘ └──────────┘ └──────────────┘
│ │
└─────┬──────┘
│
用户请求 (WebSocket 支持 /app/_stcore/stream)



### 核心流程
1. **文档加载** → `DirectoryLoader` + `TextLoader`
2. **文本分割** → `RecursiveCharacterTextSplitter` (chunk_size=400, overlap=80)
3. **向量化** → `text-embedding-v3`
4. **存储** → Chroma (持久化至宿主机目录)
5. **检索** → 余弦相似度 Top-K 检索
6. **生成** → `qwen-plus` 结合上下文与对话历史生成答案

---

## 🛠️ 技术栈

| 类型 | 技术/库 |
|------|---------|
| **框架** | LangChain 1.2, FastAPI, Streamlit |
| **LLM** | qwen-plus (可替换) |
| **Embedding** | text-embedding-v3 (1536维) |
| **向量存储** | Chroma (持久化 + HNSW 索引) |
| **日志 & 埋点** | structlog + LangChain Callbacks |
| **容器化** | Docker, Docker Compose, Nginx (反向代理) |
| **语言** | Python 3.10+ |
| **依赖管理** | pip-tools (pip-compile, pip-sync) |
---

## 🚀 快速开始

### 前置要求
- Docker & Docker Compose (推荐)
- OpenAI API Key (设置环境变量 `OPENAI_API_KEY`)

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot

2. 配置环境变量

创建 .env 文件，填入你的 API Key：
text

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
MODEL_NAME=qwen-plus

3. 安装依赖
使用 pip-sync 命令，它会根据锁定文件 requirements.txt 
精确安装所有依赖，并自动清理环境中多余的包

4. 一键启动
bash

docker compose up -d --build

服务启动后：

    前端：http://localhost/app

    API 文档：http://localhost/api/docs (Swagger)

    健康检查：http://localhost/health

5. 测试问答

    浏览器访问 http://localhost/app。

    点击左侧「知识库管理」上传一个 .md 或 .txt 文件。

    在聊天框输入基于该文档的问题，即可获得答案。

📡 API 参考
POST /api/chat

请求体：
json

{
  "question": "大语言模型通常包含多少参数？",
  "session_id": "my-session"
}

响应：
json

{
  "answer": "大语言模型通常包含数十亿甚至数千亿参数...",
  "session_id": "my-session"
}

POST /api/chat/stream (SSE)

与 /chat 入参相同，返回 text/event-stream，逐 token 推送。
POST /api/upload

使用 multipart/form-data 上传 .txt / .md 文件。
响应：
json

{
  "filename": "article.md",
  "chunks": 12,
  "status": "success"
}

📊 评测结果

    ⚠️ 本章节用于展示 RAG 系统在标准化评测数据集上的表现，请在实际运行评测后填充。

指标	得分	说明
Hit Rate (Top-3)	-	相关文档被检索到的比例
MRR (平均倒数排名)	-	相关文档在检索结果中的平均排名倒数
Faithfulness (忠实度)	-	生成答案基于检索上下文的程度 (LLM-as-Judge)
平均延迟	-	端到端问答耗时（毫秒）

评测方法见 scripts/run_evaluation.py，基于 .log/ 目录下的 eval_trace 日志自动计算。
📁 目录结构
text

.
├── api/                 # FastAPI 路由、Pydantic 模型
│   ├── app.py
│   ├── chat.py
│   ├── upload.py
│   └── models.py
├── services/            # 业务逻辑层
│   ├── document_service.py
│   ├── vector_store_service.py
│   └── rag_service.py
├── utils/               # 工具层（Embedding、LLM、日志）
│   ├── embedding.py
│   ├── llm.py
│   ├── text_splitter.py
│   └── logger.py
├── callbacks/           # LangChain 回调（评测埋点）
│   └── eval_tracker.py
├── config/              # 配置（Pydantic Settings）
│   └── settings.py
├── nginx/
│   └── nginx.conf       # Nginx 反向代理配置
├── streamlit/
│   └── config.toml      # Streamlit 子路径配置
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── app.py               # Streamlit 入口
├── requirements.in      # 顶层依赖
├── requirements.txt     # 完整锁定依赖
└── .env.example

🔧 配置说明

所有可配置项均通过 config/settings.py 管理，支持环境变量覆盖：
变量	默认值	说明
OPENAI_API_KEY	-	OpenAI API 密钥（必填）
MODEL_NAME	qwen-plus	对话模型
EMBEDDING_MODEL	text-embedding-v3	Embedding 模型
CHUNK_SIZE	400	文本分割大小
CHUNK_OVERLAP	80	相邻块重叠长度
RETRIEVAL_K	3	检索返回文档数
🧪 开发说明
    依赖管理 (使用 pip-tools)
    本项目使用 `pip-tools` 来管理依赖，以保证开发和生产环境的一致性。核心文件有两个：
    *   `requirements.in`: **顶层依赖文件**，只列出项目直接依赖的包名，不锁定版本。
    *   `requirements.txt`: **锁定依赖文件**，由 `pip-compile` 自动生成，锁定了所有直接和间接依赖的精确版本。

    本地开发（不使用 Docker）：
    bash

    # 安装依赖
    pip-sync 
    # 启动 FastAPI
    uvicorn api.app:app --reload --port 8000
    # 启动 Streamlit
    streamlit run app.py --server.port 8501

    此时前端 API 地址默认为 http://localhost:8000。

    添加新的 LLM 提供商：修改 utils/llm.py 中的 get_llm() 工厂函数即可。

    评测埋点：在 Streamlit 侧边栏开启「评测模式」后，每次问答的详细信息会写入 .log/ 目录下的 JSON 文件，供离线分析。

📋 TODO / Roadmap

    实现基于 LangSmith 的大规模批量评估

    增加多用户并发支持（Redis 存储会话）

    前端支持查看引用来源高亮

    支持更多文档格式（PDF、Word）

    集成Rerank模型提升检索精度

🤝 贡献

欢迎提交 Issue 或 Pull Request。本项目遵循 MIT 开源协议。
📮 联系

如有问题或建议，请通过 GitHub Issues 联系作者。

Demo 链接：  (部署后替换为实际地址)


> 此 README 已包含完整的项目介绍、架构、部署步骤、API 说明和评测结果框架，可直接用于 GitHub 仓库。根据实际部署地址替换 Demo 链接

