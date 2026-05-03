"""
RAG 知识库问答 - Streamlit 前端（全功能集成版）
启动前请确保 FastAPI 后端已在 8000 端口运行
"""
import json

import streamlit as st
import requests
import uuid

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==================== 自定义 CSS ====================
st.markdown("""
<style>
    /* 主标题渐变 */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* 聊天容器圆角与阴影 */
    .stChatMessage {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Sidebar 标题样式 */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }

    /* 按钮圆角 */
    .stButton > button {
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== 常量 ====================
import os
# API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ==================== 初始化会话状态 ====================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown('<p class="sidebar-header">⚙️ 会话控制</p>', unsafe_allow_html=True)

    # 流式输出开关
    use_streaming = st.toggle("流式输出", value=True, help="开启后回答将逐字显示")
    eval_mode = st.toggle("评测模式", value=False)

    # 会话信息
    st.caption(f"🆔 会话 ID: `{st.session_state.session_id}`")

    # 新对话按钮
    if st.button("🆕 新对话", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 文件上传
    st.markdown('<p class="sidebar-header">📁 知识库管理</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["txt", "md"],
        accept_multiple_files=False,
        help="支持 .txt 和 .md 文件，上传后自动分块并加入知识库"
    )

    if uploaded_file is not None:
        if st.button("📤 上传到知识库", use_container_width=True):
            with st.spinner("正在处理文档..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/upload",
                        files={"file": (uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        "text/plain")},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ `{data['filename']}` 上传成功！\n\n"
                            f"分割为 {data['chunks']} 个文本块"
                        )
                    else:
                        st.error(f"❌ 上传失败：{resp.text[:200]}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ 连接后端失败：{e}")

    st.divider()
    st.caption("💡 提示：点击聊天框输入问题，支持多轮对话")

# ==================== 主界面 ====================
# 标题
st.markdown('<p class="main-title">🤖 RAG 知识库问答系统</p>', unsafe_allow_html=True)
st.caption("基于 LangChain + Chroma 的检索增强生成助手 | 支持多轮对话与文件上传")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================== 聊天输入 ====================
if prompt := st.chat_input("请输入你的问题，按回车发送..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_answer = ""
    eval_info = None  # 用于存储评测埋点数据

    if use_streaming:
         # ---------- 流式响应（手动解析 SSE） ----------
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                resp = requests.post(
                    f"{API_BASE}/api/chat/stream",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    },
                    stream=True,
                    timeout=60
                )
                if resp.status_code == 200:
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()       # 去掉 "data:" 前缀
                        if data == "[DONE]":
                            break
                        # 检测是否为评测埋点事件
                        if data.startswith("{") and "eval_trace" in data:
                            try:
                                eval_info = json.loads(data).get("eval_trace")
                            except Exception:
                                pass
                            continue                  # 不显示到界面
                        # 普通 token，追加显示
                        full_answer += data
                        placeholder.markdown(full_answer + "▌")
                    placeholder.markdown(full_answer)
                else:
                    full_answer = f"⚠️ 请求失败 ({resp.status_code})"
                    placeholder.markdown(full_answer)
            except requests.exceptions.RequestException as e:
                full_answer = f"❌ 连接错误: {e}"
                placeholder.markdown(full_answer)
    else:
        # ---------- 非流式响应 ----------
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/chat",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id
                        },
                        timeout=60
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        full_answer = data["answer"]
                        # 提取评测字段
                        if data.get("latency_ms"):
                            eval_info = {
                                "latency_ms": data["latency_ms"],
                                "token_usage": data.get("token_usage", {"total_tokens": 0}),
                                "retrieved_docs": data.get("retrieved_docs", [])
                            }
                    else:
                        full_answer = f"⚠️ 请求失败 ({resp.status_code})"
                except requests.exceptions.RequestException as e:
                    full_answer = f"❌ 无法连接到后端服务: {e}"
            st.markdown(full_answer)

    # 保存助手回复（不包含 eval 面板的额外文字）
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # 根据评测模式开关显示埋点面板
    if eval_mode and eval_info:
        with st.expander("📊 评测埋点数据", expanded=True):
            st.metric("延迟 (ms)", f"{eval_info['latency_ms']:.2f}")
            token_usage = eval_info.get("token_usage", {})
            st.metric("Token 消耗", token_usage.get("total_tokens", 0))
            docs = eval_info.get("retrieved_docs", [])
            if docs:
                st.caption("检索到的文档")
                for doc in docs:
                    st.text(f"📄 {doc['source']}: {doc['content'][:80]}...")


        """# 助手回复
        with st.chat_message("assistant"):
            if use_streaming:
                # ---------- 流式响应 ----------
                def stream_response():
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/chat/stream",
                            json={
                                "question": prompt,
                                "session_id": st.session_state.session_id
                            },
                            stream=True,
                            timeout=60
                        )
                        if resp.status_code != 200:
                            yield f"⚠️ 请求失败 ({resp.status_code})"
                            return

                        for line in resp.iter_lines(decode_unicode=True):
                            if line and line.startswith("data:"):
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    break
                                yield data
                    except requests.exceptions.RequestException as e:
                        yield f"❌ 连接错误: {e}"

                full_answer = st.write_stream(stream_response)
                if full_answer is None:
                    full_answer = ""

            else:
                # ---------- 一次性响应 ----------
                with st.spinner("🤔 思考中..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/chat",
                            json={
                                "question": prompt,
                                "session_id": st.session_state.session_id
                            },
                            timeout=60
                        )
                        if resp.status_code == 200:
                            full_answer = resp.json()["answer"]
                        else:
                            full_answer = f"⚠️ 请求失败 ({resp.status_code})"
                    except requests.exceptions.RequestException as e:
                        full_answer = f"❌ 无法连接到后端服务: {e}"
                st.markdown(full_answer)

            # 保存助手回复
            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer}
            )"""

    