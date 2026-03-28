import json
import re
from typing import Any, Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def split_into_chunks(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks = re.split(r'(?<=[.!?])\s+', text)
    chunks = [c.strip() for c in chunks if c.strip()]

    if not chunks:
        return [text]

    return chunks


def cosine_sim(vec_a, vec_b) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    return float(cosine_similarity([vec_a], [vec_b])[0][0])


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))