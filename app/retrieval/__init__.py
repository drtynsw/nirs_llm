"""Поиск по корпусу законов: предобработка + собственный BM25 (spec/tech/rag.md).

Интерфейс для сервера тот же, что был у заглушки КТ1:
    search(query: str, top_k: int) -> list[dict]
плюс answerable(...) — порог «нет ответа» (сценарий С4).
"""
import threading
from dataclasses import asdict, dataclass

from app.retrieval import preprocess
from app.retrieval.bm25 import BM25Index
from app.retrieval.corpus import load_corpus
from app.retrieval.expansion import expand


@dataclass(frozen=True)
class Config:
    mode: str = "lemma"        # none | stem | lemma
    k1: float = 2.0
    b: float = 0.75
    expand: bool = True
    title_weight: int = 3      # сколько раз повторяем название статьи в документе
    min_score: float = 10.0    # порог С4: оценка BM25 лучшего документа

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mode_effective"] = preprocess.resolve_mode(self.mode)
        return d


DEFAULT = Config()


class Retriever:
    def __init__(self, config: Config = DEFAULT):
        self.config = config
        self.mode = preprocess.resolve_mode(config.mode)
        self.corpus = {d["doc_id"]: d for d in load_corpus()}
        # Индексируем название вместе с текстом: в названии часто стоит ключевой термин.
        docs = {i: preprocess.terms((d["title"] + " ") * config.title_weight + d["text"], self.mode)
                for i, d in self.corpus.items()}
        self.index = BM25Index(docs, k1=config.k1, b=config.b)

    def query_terms(self, query: str) -> list[str]:
        q = preprocess.terms(query, self.mode)
        if self.config.expand:
            q += expand(query, self.mode)
        return q

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.index.search(self.query_terms(query), top_k)

    def answerable(self, top_score: float) -> bool:
        """С4: лучший документ набрал достаточную оценку BM25."""
        return top_score >= self.config.min_score

    def snippet(self, text: str, query_terms: set[str], width: int = 280) -> str:
        flat = " ".join(text.split())
        low = flat.lower().replace("ё", "е")
        pos = 0
        for m in preprocess.TOKEN.finditer(low):
            if preprocess.normalize(m.group(), self.mode) in query_terms:
                pos = m.start()
                break
        start = max(0, pos - 60)
        if start:
            space = flat.find(" ", start)
            start = space + 1 if 0 <= space < pos else start
        cut = flat[start:start + width]
        return ("…" if start else "") + cut + ("…" if start + width < len(flat) else "")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = self.query_terms(query)
        ranked = self.index.search(q, top_k)
        qset = set(q)
        out = []
        for doc_id, score in ranked:
            d = self.corpus[doc_id]
            out.append({
                "doc_id": doc_id, "citation": d["citation"], "law": d["law"],
                "article": d["article"], "title": d["title"], "score": round(score, 3),
                "snippet": self.snippet(d["text"], qset),
            })
        return out


_lock = threading.Lock()
_cache: dict[Config, Retriever] = {}


def get(config: Config = DEFAULT) -> Retriever:
    """Индекс строится один раз на конфигурацию (≈1–2 с для 513 документов)."""
    with _lock:
        if config not in _cache:
            _cache[config] = Retriever(config)
        return _cache[config]


def search(query: str, top_k: int = 5) -> list[dict]:
    return get().search(query, top_k)
