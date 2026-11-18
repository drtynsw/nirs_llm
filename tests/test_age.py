"""Учёт возраста (С6). Запуск: python -m unittest -v"""
import unittest

from app import age, analytics, server

server.STORE = analytics.Store(":memory:")


class ValidateTest(unittest.TestCase):
    def test_missing_age_is_none(self):
        self.assertIsNone(age.validate(None))
        self.assertIsNone(age.validate(""))

    def test_valid_ages(self):
        self.assertEqual(age.validate(17), 17)
        self.assertEqual(age.validate("16"), 16)
        self.assertEqual(age.validate(18.0), 18)

    def test_invalid_ages(self):
        for raw in (9, 101, "abc", 16.5, True, [17]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                age.validate(raw)

    def test_groups(self):
        cases = {10: "до 14", 13: "до 14", 14: "14–15", 15: "14–15", 16: "16–17",
                 17: "16–17", 18: "18–23", 23: "18–23", 24: "24+", 60: "24+", None: None}
        for value, expected in cases.items():
            self.assertEqual(age.group(value), expected)


class AskWithAgeTest(unittest.TestCase):
    def test_bad_age_rejected(self):
        code, body = server.ask({"question": "отпуск", "age": 5})
        self.assertEqual(code, 400)
        self.assertIn("Возраст", body["error"])

    def test_without_age_fields_present_and_empty(self):
        _, body = server.ask({"question": "отпуск"})
        self.assertIsNone(body["age_group"])
        self.assertEqual(body["age_documents"], [])

    def test_adult_answer_unchanged(self):
        _, plain = server.ask({"question": "Можно ли работать ночью?"})
        _, adult = server.ask({"question": "Можно ли работать ночью?", "age": 20})
        self.assertEqual(plain["answer"], adult["answer"])
        self.assertEqual(adult["age_group"], "18–23")
        self.assertEqual(adult["age_documents"], [])

    def test_minor_gets_note_and_articles(self):
        _, body = server.ask({"question": "Можно ли меня ставить на работу в ночное время?", "age": 16})
        self.assertEqual(body["age_group"], "16–17")
        self.assertIn("Вам 16 лет", body["answer"])
        self.assertIn("ст. 268 ТК РФ", body["answer"])
        self.assertTrue(body["answer"].endswith("не юридическая консультация."))
        ids = [d["doc_id"] for d in body["age_documents"]]
        self.assertTrue(ids)
        self.assertLessEqual(len(ids), 3)
        # ст. 268 — либо уже в выдаче поиска, либо в блоке «Важно в вашем возрасте»
        shown = ids + [d["doc_id"] for d in body["documents"]]
        self.assertIn("tk_268", shown)

    def test_age_documents_do_not_repeat_search_results(self):
        _, body = server.ask({"question": "отпуск работникам в возрасте до восемнадцати лет", "age": 17})
        found = {d["doc_id"] for d in body["documents"]}
        self.assertFalse(found & {d["doc_id"] for d in body["age_documents"]})

    def test_young_teen_gets_article_63(self):
        _, body = server.ask({"question": "Как устроиться на подработку?", "age": 14})
        self.assertIn("ст. 63 ТК РФ", body["answer"])
        self.assertEqual(body["age_documents"][0]["doc_id"], "tk_63")

    def test_age_does_not_change_search_ranking(self):
        _, plain = server.ask({"question": "сверхурочная работа", "top_k": 5})
        _, minor = server.ask({"question": "сверхурочная работа", "top_k": 5, "age": 15})
        self.assertEqual([d["doc_id"] for d in plain["documents"]],
                         [d["doc_id"] for d in minor["documents"]])


if __name__ == "__main__":
    unittest.main()
