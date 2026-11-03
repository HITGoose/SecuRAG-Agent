"""
本地 trace 日志（阶段5）：每个用户请求一条 JSON，追加写到 logs/traces-YYYYMMDD.jsonl（logs/ 不进 git）。

记录：意图、路由、每次模型调用（用途、实际后端、token、耗时、重试、是否降级）、
每个工具调用（名字、模型给的参数、状态、错误码、走本地还是 MCP、耗时）、事实依据检查、端到端耗时。
只记脱敏后的内容：用户消息和回答都是出口脱敏后的版本；会话 id 做哈希；不记系统提示、密钥、异常栈。
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"


def anon(session_id: str) -> str:
    return hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:10]


def build_trace(trace_id: str, session_id: str, user_id: str, safe_msg: str, out: dict, llm_log: list) -> dict:
    return {
        "trace_id": trace_id,
        "time": datetime.now().isoformat(timespec="seconds"),
        "session": anon(session_id),
        "user": user_id,  # 演示用户 ID（USR-DEMO-*），不是真实身份
        "message": safe_msg[:300],
        "route": out["route"], "intent": out["intent"], "stop_reason": out["stop_reason"],
        "fact_check": out.get("fact_check"),
        "latency_ms": out["latency_ms"],
        "llm_calls": len(llm_log),
        "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
        "tokens_complete": out["tokens_complete"],
        "fallback_used": any(c.get("fallback_reason") for c in llm_log),
        "llm": llm_log,
        "tools": [{k: s.get(k) for k in ("step", "tool", "args", "status", "error_code", "via", "latency_ms")}
                  for s in out["steps"]],
        "answer": out["answer"][:500],
    }


class TraceWriter:
    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)

    def write(self, record: dict) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"traces-{datetime.now():%Y%m%d}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path


def read_traces(log_dir: Path = LOG_DIR) -> list:
    records = []
    for path in sorted(Path(log_dir).glob("traces-*.jsonl")):
        with path.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return records
