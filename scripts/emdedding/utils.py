# =============================================================================
# utils.py — Shared utility functions for the embedding-based pipeline
#
#   - load_json       : load a standard JSON file
#   - load_jsonl      : load a newline-delimited JSON file (JSONL)
#   - normalize_text  : strip and collapse whitespace in a string
#   - split_into_chunks : sentence-level tokenization for chunked similarity
#   - cosine_sim      : cosine similarity between two embedding vectors
#   - clamp           : clamp a float value within [low, high]
# =============================================================================

import json
import re
from typing import Any, Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity


def load_json(path: str) -> Any:
    # Load and return the contents of a standard JSON file.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    # Load a JSONL file and return a list of parsed objects.
    # Empty lines are skipped silently.
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text: Optional[str]) -> str:
    # Normalize a string: handle None, strip leading/trailing whitespace,
    # and collapse consecutive whitespace characters into a single space.
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def split_into_chunks(text: str) -> List[str]:
    # Split a response into sentence-level chunks using punctuation boundaries
    # (.  !  ?) as delimiters. Used to enable worst-case chunk similarity scoring.
    # Falls back to returning the full text as a single chunk if no splits are found.
    text = normalize_text(text)
    if not text:
        return []

    chunks = re.split(r'(?<=[.!?])\s+', text)
    chunks = [c.strip() for c in chunks if c.strip()]

    # Fallback: return the whole text if no sentence boundary was found
    if not chunks:
        return [text]

    return chunks


def cosine_sim(vec_a, vec_b) -> float:
    # Compute cosine similarity between two embedding vectors.
    # Returns 0.0 if either vector is None (safe fallback).
    if vec_a is None or vec_b is None:
        return 0.0
    return float(cosine_similarity([vec_a], [vec_b])[0][0])


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    # Clamp a float value to the range [low, high].
    # Used to ensure all scores stay within valid bounds.
    return max(low, min(high, value))