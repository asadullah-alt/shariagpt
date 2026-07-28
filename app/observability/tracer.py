import json
import time
from pathlib import Path
from pydantic import BaseModel


class TraceRecord(BaseModel):
    request_id: str
    session_id: str
    timestamp: str
    latency_ms: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    chunk_ids: list[str]
    relevance_scores: list[float]
    avg_relevance_score: float
    pii_detected: list[str]
    query_length: int
    response_length: int
    cache_hit: bool = False


def emit_trace(record: TraceRecord, log_dir: str = "logs") -> None:
    """Write trace as JSON line to file and stdout."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json()
    with open(Path(log_dir) / "traces.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"TRACE {line}", flush=True)
