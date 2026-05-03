# utils/logger.py
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

import structlog

class EvaluationLogger:
    """
    评测埋点日志管理器
    - 控制台：输出 JSON 格式日志 (便于开发观察)
    - 文件：写入 .log/ 目录下的 JSON 日志文件，每行一条 JSON 记录
    """

    def __init__(
        self,
        log_dir: str = "./.log",
        app_name: str = "rag-system",
        json_console: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id: Optional[str] = None

        # 确保标准库日志能输出 INFO 及以上级别到控制台
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(message)s',
                handlers=[logging.StreamHandler()]
            )

        # 配置 structlog 处理器管道
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer() if json_console
                else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger(app_name)

        # 文件日志处理器（独立，每行一个 JSON）
        self._file_logger = logging.getLogger(f"{app_name}.file")
        self._file_logger.setLevel(logging.DEBUG)
        log_file = self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(
            self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter('%(message)s'))
        self._file_logger.addHandler(fh)
        # ✅ 关键：立即写入一条测试记录，确保文件可写
        self._file_logger.info(json.dumps({"event": "log_init", "status": "ok", "file": str(log_file)}))
        print(f"✅ 日志文件初始化成功: {log_file}")   # 控制台提示
    
    def set_session(self, session_id: str):
        self._session_id = session_id
        self.logger = self.logger.bind(session_id=session_id)

    # ---------- 评测埋点 API ----------
    def log_eval_trace(
        self,
        question: str,
        answer: Optional[str],
        retrieved_docs: list,
        latency_ms: float,
        token_usage: Dict[str, int],
        hit_rate: Optional[float] = None,
        mrr: Optional[float] = None,
    ):
        """记录一次完整的问答评测追踪"""
        payload: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "retrieved_doc_count": len(retrieved_docs),
            "retrieved_sources": [d.metadata.get("source", "") for d in retrieved_docs],
            "latency_ms": round(latency_ms, 2),
            "token_usage": token_usage,
            "hit_rate": hit_rate,
            "mrr": mrr,
        }
        self.logger.info("eval_trace", **payload)                 # 控制台输出
        # 文件日志：手动构造包含 event 的完整字典
        file_data = {"event": "eval_trace", **payload}
        self._file_logger.info(json.dumps(file_data, ensure_ascii=False))

    def log_llm_call(self, model: str, prompt_tokens: int, duration_ms: float):
        """记录一次 LLM 调用（使用 INFO 级别 + 特定 event）"""
        self.logger.info(
            "llm_call",
            model=model,
            prompt_tokens=prompt_tokens,
            duration_ms=round(duration_ms, 2),
        )

    def log_retrieval(self, query: str, doc_count: int, duration_ms: float):
        """记录一次向量检索"""
        self.logger.debug(
            "retrieval",
            query=query[:100],
            doc_count=doc_count,
            duration_ms=round(duration_ms, 2),
        )

    def info(self, event: str, **kwargs):
        self.logger.info(event, **kwargs)

    def error(self, event: str, **kwargs):
        self.logger.error(event, **kwargs)


# 全局单例
_eval_logger: Optional[EvaluationLogger] = None

def get_eval_logger() -> EvaluationLogger:
    global _eval_logger
    if _eval_logger is None:
        _eval_logger = EvaluationLogger()
    return _eval_logger


# 简单自我测试
if __name__ == "__main__":
    logger = get_eval_logger()
    logger.set_session("test-123")
    logger.info("app_start", version="1.0")
    logger.log_llm_call("gpt-4.1-mini", 200, 1234.5)
    print("✅ 日志输出测试完成，请检查控制台和 .log/ 目录")