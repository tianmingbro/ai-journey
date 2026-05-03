# callbacks/eval_tracker.py
import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from utils.logger import get_eval_logger


class TimedContext:
    """轻量级计时上下文管理器，用于精确测量代码块耗时"""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


class EvaluationTracker(BaseCallbackHandler):
    """
    评测埋点追踪器。
    在 LLM 调用、检索、工具使用等关键节点自动记录耗时与结果，
    最终通过 flush_trace() 输出一条完整的 eval_trace 日志。
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.logger = get_eval_logger()
        self.logger.set_session(session_id)

        self._timers: Dict[str, TimedContext] = {}
        self._retrieved_docs: List[Document] = []
        self._question: Optional[str] = None
        self._answer: Optional[str] = None
        self._total_tokens: int = 0
        self._model_name: str = "unknown"

    # ---------- LLM 埋点 ----------
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self._timers["llm"] = TimedContext()
        self._timers["llm"].__enter__()
        # 提取模型名称
        self._model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        # 记录问题（取最后一条 human 消息）
        self._question = prompts[-1][:200] if prompts else ""

    def on_llm_end(self, response, **kwargs):
        if "llm" in self._timers:
            self._timers["llm"].__exit__()
            duration_ms = self._timers["llm"].elapsed_ms
        else:
            duration_ms = 0.0

        # 提取 token 用量
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self._total_tokens = usage.get("total_tokens", 0)

        # 提取生成文本
        if hasattr(response, "generations") and response.generations:
            first_gen = response.generations[0][0]
            if hasattr(first_gen, "message"):
                self._answer = first_gen.message.content

        self.logger.log_llm_call(self._model_name, self._total_tokens, duration_ms)

    # ---------- 检索埋点 ----------
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        self._timers["retrieval"] = TimedContext()
        self._timers["retrieval"].__enter__()

    def on_retriever_end(self, documents: List[Document], **kwargs):
        if "retrieval" in self._timers:
            self._timers["retrieval"].__exit__()
            duration_ms = self._timers["retrieval"].elapsed_ms
        else:
            duration_ms = 0.0

        self._retrieved_docs = documents
        self.logger.log_retrieval(
            self._question or "", len(documents), duration_ms
        )

    # ---------- 工具埋点 ----------
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        name = serialized.get("name", "unknown")
        self._timers[f"tool:{name}"] = TimedContext()
        self._timers[f"tool:{name}"].__enter__()

    def on_tool_end(self, output: str, **kwargs):
        # 关闭最后一个未完成的 tool 计时器
        for key in list(self._timers.keys()):
            if key.startswith("tool:") and not hasattr(self._timers[key], "elapsed_ms"):
                self._timers[key].__exit__()
                break

    # ---------- 汇总埋点（手动调用） ----------
    def flush_trace(self, question: str, answer: str, docs: List[Document]):
        """在业务逻辑调用结束后，将完整追踪信息写入日志"""
        total_latency = sum(
            timer.elapsed_ms for timer in self._timers.values()
            if hasattr(timer, "elapsed_ms")
        )
        self.logger.log_eval_trace(
            question=question,
            answer=answer,
            retrieved_docs=docs,
            latency_ms=total_latency,
            token_usage={"total_tokens": self._total_tokens},
        )