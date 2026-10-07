"""Тесты прототипа (КТ1). Запуск: python -m unittest -v"""
import unittest

from app import server


class AskTest(unittest.TestCase):
    def test_empty_question_rejected(self):
        code, body = server.ask({"question": "   "})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_too_long_question_rejected(self):
        code, _ = server.ask({"question": "а" * 501})
        self.assertEqual(code, 400)

    def test_answer_has_contract_fields(self):
        code, body = server.ask({"question": "Сколько часов может работать студент?", "top_k": 3})
        self.assertEqual(code, 200)
        for key in ("question", "answer", "mode", "documents", "took_ms"):
            self.assertIn(key, body)
        self.assertLessEqual(len(body["documents"]), 3)
        doc = body["documents"][0]
        for key in ("doc_id", "citation", "law", "article", "title", "score", "snippet"):
            self.assertIn(key, doc)

    def test_top_k_is_clamped(self):
        _, body = server.ask({"question": "отпуск", "top_k": 100})
        self.assertLessEqual(len(body["documents"]), 10)

    def test_answer_has_disclaimer(self):
        _, body = server.ask({"question": "увольнение по собственному желанию"})
        self.assertIn("не юридическая консультация", body["answer"])

    def test_relevant_article_found(self):
        _, body = server.ask({"question": "отпуск без сохранения заработной платы", "top_k": 5})
        self.assertIn("128", [d["article"] for d in body["documents"]])


class CorpusTest(unittest.TestCase):
    def test_corpus_loaded(self):
        stats = server.sources()
        self.assertGreater(stats["documents"], 400)
        self.assertEqual({s["law"] for s in stats["sources"]},
                         {"ТК РФ", "ГК РФ", "422-ФЗ (самозанятые)"})


if __name__ == "__main__":
    unittest.main()
