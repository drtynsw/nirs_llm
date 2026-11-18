"""Собственный инвертированный индекс и ранжирование BM25 (методичка НИРС, разд. 1.3–1.4).

    IDF(t)     = ln(1 + (N − DF(t) + 0.5) / (DF(t) + 0.5))
    BM25(q, d) = Σ_{t∈q} IDF(t) · TF(t,d)·(k1 + 1) / (TF(t,d) + k1·(1 − b + b·|d|/avgdl))

Готовые поисковые индексы не используются (условие НИРС).
"""
from collections import Counter, defaultdict
from math import log


class BM25Index:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        """docs: doc_id -> список термов документа (уже после предобработки)."""
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)   # терм -> {doc_id: TF}
        self.doc_len: dict[str, int] = {}
        for doc_id, words in docs.items():
            self.doc_len[doc_id] = len(words)
            for term, tf in Counter(words).items():
                self.postings[term][doc_id] = tf
        self.postings = dict(self.postings)
        self.N = len(docs)
        self.avgdl = sum(self.doc_len.values()) / self.N if self.N else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def idf(self, term: str) -> float:
        df = self.df(term)
        return log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score_all(self, query_terms: list[str]) -> dict[str, float]:
        """Оценки всех документов, где есть хотя бы один терм запроса.

        Повтор терма в запросе учитывается один раз (запросы короткие).
        """
        scores: dict[str, float] = defaultdict(float)
        for term in dict.fromkeys(query_terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query_terms: list[str], top_k: int) -> list[tuple[str, float]]:
        """top_k пар (doc_id, оценка) по убыванию оценки; равные — по doc_id."""
        scores = self.score_all(query_terms)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [(d, s) for d, s in ranked[:top_k] if s > 0]
