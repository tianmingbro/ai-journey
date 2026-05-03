# test_eval_tracker.py
import os
from dotenv import load_dotenv
load_dotenv()

from services.rag_service import RAGService
from utils.logger import get_eval_logger

if __name__ == "__main__":
    logger = get_eval_logger()
    rag = RAGService()

    session = "eval-test"
    q = "大语言模型通常包含多少参数？"
    print(f"Question: {q}")
    ans = rag.chat(session, q)
    print(f"Answer: {ans}\n")

    # 检查日志文件
    log_dir = logger.log_dir
    latest_log = sorted(log_dir.glob("*.log"))[-1]
    with open(latest_log, "r") as f:
        lines = f.readlines()
        eval_lines = [l for l in lines if "eval_trace" in l]
        print(f"在 {latest_log} 中找到 {len(eval_lines)} 条 eval_trace 日志")