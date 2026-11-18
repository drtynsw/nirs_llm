"""Загрузка корпуса data/corpus.jsonl (собирается scripts/build_corpus.py)."""
import json
from functools import lru_cache
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent.parent / "data" / "corpus.jsonl"


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    with CORPUS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id() -> dict[str, dict]:
    return {d["doc_id"]: d for d in load_corpus()}
