"""Предобработка, BM25 и benchmark. Запуск: python -m unittest -v"""
import math
import unittest

from app import rag, retrieval
from app.retrieval import preprocess
from app.retrieval.bm25 import BM25Index
from app.retrieval.stemmer import stem
from benchmark import run as bench


class StemmerTest(unittest.TestCase):
    def test_snowball_reference_forms(self):
        # Эталонные значения стеммера Snowball для русского языка.
        cases = {"уволили": "увол", "уволить": "увол", "отпуска": "отпуск", "отпуском": "отпуск",
                 "работодателя": "работодател", "трудового": "трудов", "увольнения": "увольнен",
                 "красивейший": "красив"}
        for word, expected in cases.items():
            self.assertEqual(stem(word), expected, word)


class PreprocessTest(unittest.TestCase):
    def test_tokens_keep_article_numbers_and_drop_punctuation(self):
        self.assertEqual(preprocess.raw_tokens("Статья 312.1. Дистанционная работа!"),
                         ["статья", "312.1", "дистанционная", "работа"])

    def test_yo_and_stopwords(self):
        self.assertEqual(preprocess.terms("Можно ли мне её уволить?", "none"), ["уволить"])

    def test_same_lemma_for_word_forms(self):
        mode = preprocess.resolve_mode("lemma")
        forms = {preprocess.terms(w, mode)[0] for w in ("отпуск", "отпуска", "отпуском")}
        self.assertEqual(len(forms), 1)


class BM25Test(unittest.TestCase):
    """Пример из занятия 3: три документа по 4 токена, запрос «как происходит возврат оплаты»."""

    def setUp(self):
        self.index = BM25Index({"A": ["возврат", "оплаты", "на", "карту"],
                                "B": ["статус", "оплаты", "в", "приложении"],
                                "C": ["возврат", "билета", "через", "кассу"]}, k1=1.2, b=0.75)
        self.query = ["как", "происходит", "возврат", "оплаты"]

    def test_idf_from_lecture(self):
        self.assertAlmostEqual(self.index.idf("возврат"), math.log(1 + 1.5 / 2.5), places=6)
        self.assertAlmostEqual(self.index.idf("возврат"), 0.470, places=3)

    def test_scores_from_lecture(self):
        scores = self.index.score_all(self.query)
        self.assertAlmostEqual(scores["A"], 0.940, places=3)
        self.assertAlmostEqual(scores["B"], 0.470, places=3)
        self.assertAlmostEqual(scores["C"], 0.470, places=3)

    def test_ties_by_id_and_top_k(self):
        self.assertEqual([d for d, _ in self.index.search(self.query, 2)], ["A", "B"])

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.search(["погода"], 5), [])

    def test_repeated_query_term_counted_once(self):
        self.assertEqual(self.index.score_all(["возврат"]), self.index.score_all(["возврат", "возврат"]))


class RetrieverTest(unittest.TestCase):
    def test_relevant_articles_found(self):
        cases = {"отпуск без сохранения заработной платы": "tk_128",
                 "максимальный срок испытания при приеме на работу": "tk_70",
                 "нужно ли самозанятому сдавать декларацию": "npd_13"}
        for query, doc_id in cases.items():
            ids = [d["doc_id"] for d in retrieval.search(query, 5)]
            self.assertIn(doc_id, ids, query)

    def test_scores_sorted(self):
        scores = [d["score"] for d in retrieval.search("задерживают зарплату", 10)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_off_topic_refused(self):
        result = rag.respond("какая погода будет в Москве на выходных", 5)
        self.assertTrue(result["refused"])
        self.assertEqual(result["documents"], [])

    def test_stem_fallback_works(self):
        r = retrieval.get(retrieval.Config(mode="stem"))
        self.assertIn("tk_70", [d["doc_id"] for d in r.search("испытательный срок", 5)])


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = bench.load_queries()
        cls.report = bench.evaluate()

    def test_dataset_follows_course_rules(self):
        regular = [q for q in self.queries if q["type"] == "regular"]
        traps = [q for q in self.queries if q["type"] == "trap"]
        self.assertGreaterEqual(len(regular), 10)
        self.assertGreaterEqual(len(traps), 5)
        for q in regular:
            self.assertGreaterEqual(len(q["relevant"]), 3, q["id"])
            self.assertTrue(q["reference"], q["id"])

    def test_relevant_ids_exist(self):
        ids = {d["doc_id"] for d in retrieval.corpus.load_corpus()}
        for q in self.queries:
            for r in q["relevant"]:
                self.assertTrue(any(bench.matches(i, r) for i in ids), f"{q['id']}: {r}")

    def test_article_level_matching(self):
        self.assertTrue(bench.matches("tk_81_2", "tk_81"))
        self.assertFalse(bench.matches("tk_80", "tk_8"))

    def test_metrics_formulas(self):
        self.assertAlmostEqual(bench.recall_at(["B", "A", "D"], ["A", "D", "F"], 3), 2 / 3)
        self.assertEqual(bench.first_rank(["B", "A", "D"], ["A", "D", "F"]), 2)
        self.assertIsNone(bench.first_rank(["B"], ["A"]))

    def test_report_has_four_metrics(self):
        m = self.report["metrics"]
        for key in ("recall@5", "recall@10", "mrr@10", "correct_rejection_rate", "accuracy"):
            self.assertIn(key, m)

    def test_quality_not_below_recorded_baseline(self):
        """Регрессия: метрики не хуже зафиксированных на КТ2 (spec/tech/rag.md)."""
        m = self.report["metrics"]
        self.assertGreaterEqual(m["recall@5"], 0.40)
        self.assertGreaterEqual(m["mrr@10"], 0.70)
        self.assertGreaterEqual(m["correct_rejection_rate"], 0.6)
        self.assertLessEqual(m["false_rejection_rate"], 0.05)


if __name__ == "__main__":
    unittest.main()
