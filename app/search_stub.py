"""ВРЕМЕННЫЙ поиск для прототипа (контрольная точка 1).

Считает, сколько слов запроса встречается в статье. Нужен только для того,
чтобы интерфейс показывал настоящие статьи корпуса. На КТ2 заменяется
модулем app/retrieval (BM25, см. spec/tech/rag.md) с тем же интерфейсом:
    search(query: str, top_k: int) -> list[dict]
"""
import json
import re
from functools import lru_cache
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"
WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    with CORPUS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _words(text: str) -> list[str]:
    # Грубая «основа»: первые 5 букв. Только для прототипа, не для бенчмарка.
    return [w.lower()[:5] for w in WORD.findall(text) if len(w) > 2]


def snippet(text: str, query_words: set[str], width: int = 280) -> str:
    flat = " ".join(text.split())
    low = flat.lower()
    pos = min((low.find(w) for w in query_words if low.find(w) >= 0), default=0)
    start = max(0, pos - 60)
    cut = flat[start:start + width]
    return ("…" if start else "") + cut + ("…" if start + width < len(flat) else "")


def search(query: str, top_k: int = 5) -> list[dict]:
    q = set(_words(query))
    if not q:
        return []
    scored = []
    for doc in load_corpus():
        words = _words(doc["title"] + " " + doc["text"])
        score = sum(1 for w in words if w in q)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "doc_id": d["doc_id"],
            "citation": d["citation"],
            "law": d["law"],
            "article": d["article"],
            "title": d["title"],
            "score": float(s),
            "snippet": snippet(d["text"], q),
        }
        for s, d in scored[:top_k]
    ]
