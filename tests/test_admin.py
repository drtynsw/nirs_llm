"""Роль администратора и аналитика (С7). Запуск: python -m unittest -v"""
import json
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

from app import analytics, server

TOKEN = "test-admin-token"
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
DOC = {"doc_id": "tk_80", "citation": "ст. 80 ТК РФ"}


class TopicTest(unittest.TestCase):
    def test_topics(self):
        self.assertEqual(analytics.topic("Как уволиться по собственному желанию?"), "Увольнение")
        self.assertEqual(analytics.topic("Какой налог платит самозанятый?"), "Самозанятость")
        self.assertEqual(analytics.topic("Какая погода в Москве?"), "Другое")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = analytics.Store(":memory:")
        self.addCleanup(self.store.close)

    def test_empty_stats(self):
        s = self.store.stats(7, now=NOW)
        self.assertEqual(s["total"], 0)
        self.assertEqual(len(s["by_day"]), 7)
        self.assertEqual(s["by_age_group"][-1], {"group": "не указан", "count": 0})

    def test_aggregates(self):
        rec = self.store.record
        rec(question="хочу уволиться", age_group="16–17", top_k=5, documents=[DOC], took_ms=10, now=NOW)
        rec(question="хочу уволиться", age_group=None, top_k=5, documents=[DOC], took_ms=20, now=NOW)
        rec(question="погода", age_group="18–23", top_k=3, documents=[], took_ms=30,
            now=NOW - timedelta(days=1))
        rec(question="старое", age_group=None, top_k=5, documents=[], took_ms=1,
            now=NOW - timedelta(days=40))
        s = self.store.stats(30, now=NOW)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["no_answer"], 1)
        self.assertEqual(s["avg_took_ms"], 20.0)
        self.assertAlmostEqual(s["age_given_share"], 0.667)
        self.assertEqual(s["by_day"][-1], {"date": "2026-11-18", "count": 2})
        self.assertEqual(s["by_day"][-2]["count"], 1)
        self.assertEqual(s["by_topic"][0], {"topic": "Увольнение", "count": 2})
        self.assertEqual(s["top_articles"], [{"doc_id": "tk_80", "citation": "ст. 80 ТК РФ", "count": 2}])
        self.assertIn({"group": "16–17", "topic": "Увольнение", "count": 1}, s["age_topic"])

    def test_question_text_not_stored(self):
        self.store.record(question="секретная ситуация с ООО Ромашка", age_group="16–17", top_k=5,
                          documents=[DOC], took_ms=5, now=NOW)
        dump = "\n".join(self.store._db.iterdump())
        self.assertNotIn("Ромашка", dump)
        self.assertNotIn("Ромашка", self.store.export_csv(30, now=NOW))

    def test_csv(self):
        self.store.record(question="уволиться", age_group="16–17", top_k=5, documents=[DOC],
                          took_ms=5, now=NOW)
        lines = self.store.export_csv(30, now=NOW).lstrip("﻿").splitlines()
        self.assertEqual(lines[0], ";".join(analytics.CSV_FIELDS))
        self.assertIn("16–17;Увольнение;5;1;tk_80", lines[1])


class AdminApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = server.ADMIN_TOKEN, server.STORE
        server.ADMIN_TOKEN = TOKEN
        server.STORE = analytics.Store(":memory:")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.httpd.log_message = lambda *a: None
        server.Handler.log_message = lambda *a: None
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        server.STORE.close()
        server.ADMIN_TOKEN, server.STORE = cls.saved

    def _get(self, path, token=None):
        req = urllib.request.Request(self.base + path)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()

    def _ask(self, payload):
        req = urllib.request.Request(self.base + "/api/ask", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def test_stats_require_token(self):
        self.assertEqual(self._get("/api/admin/stats")[0], 401)
        self.assertEqual(self._get("/api/admin/stats", "wrong")[0], 401)
        self.assertEqual(self._get("/api/admin/export.csv")[0], 401)

    def test_bad_days(self):
        self.assertEqual(self._get("/api/admin/stats?days=0", TOKEN)[0], 400)
        self.assertEqual(self._get("/api/admin/stats?days=abc", TOKEN)[0], 400)

    def test_ask_is_recorded(self):
        before = json.loads(self._get("/api/admin/stats?days=7", TOKEN)[2])["total"]
        body = self._ask({"question": "Хочу уволиться, а не отпускают", "age": 17})
        self.assertEqual(body["age_group"], "16–17")
        code, headers, raw = self._get("/api/admin/stats?days=7", TOKEN)
        self.assertEqual(code, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        stats = json.loads(raw)
        self.assertEqual(stats["total"], before + 1)
        self.assertNotIn("отпускают", raw.decode("utf-8"))

    def test_csv_export(self):
        code, headers, raw = self._get("/api/admin/export.csv?days=7", TOKEN)
        self.assertEqual(code, 200)
        self.assertIn("text/csv", headers["Content-Type"])
        self.assertTrue(raw.decode("utf-8").lstrip("﻿").startswith("ts;age_group"))

    def _post(self, path, payload, token=None):
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_benchmark_requires_token(self):
        self.assertEqual(self._get("/api/admin/benchmark")[0], 401)
        self.assertEqual(self._post("/api/admin/benchmark", {})[0], 401)

    def test_body_limit(self):
        from http.client import HTTPConnection

        for path in ("/api/ask", "/api/admin/benchmark"):
            for size, code in ((server.MAX_BODY_BYTES, 200), (server.MAX_BODY_BYTES + 1, 413)):
                with self.subTest(path=path, size=size):
                    body = b'{"question":"test"}'.ljust(size)
                    conn = HTTPConnection(*self.httpd.server_address)
                    try:
                        conn.request("POST", path, body, {"Authorization": f"Bearer {TOKEN}"})
                        response = conn.getresponse()
                        self.assertEqual(response.status, code)
                        self.assertIsInstance(json.loads(response.read()), dict)
                    finally:
                        conn.close()

    def test_parallel_benchmark_rejected(self):
        started, finish = threading.Event(), threading.Event()

        def evaluate(*args, **kwargs):
            started.set()
            if not finish.wait(5):
                raise TimeoutError("Тестовый прогон не завершился")
            return {"done": True}

        with patch.object(server.bench, "evaluate", side_effect=evaluate) as run:
            results = []
            thread = threading.Thread(target=lambda: results.append(self._post("/api/admin/benchmark", {}, TOKEN)))
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                code, error = self._post("/api/admin/benchmark", {}, TOKEN)
                self.assertEqual(code, 409)
                self.assertIn("error", error)
                self.assertEqual(self._get("/api/health")[0], 200)
            finally:
                finish.set()
                thread.join(5)
            self.assertEqual(results, [(200, {"done": True})])
            run.assert_called_once()

    def test_benchmark_unlocks_after_error(self):
        with patch.object(server.bench, "evaluate", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                server.admin_benchmark(f"Bearer {TOKEN}", {})
        with patch.object(server.bench, "evaluate", return_value={}):
            self.assertEqual(server.admin_benchmark(f"Bearer {TOKEN}", {}), (200, {}))

    def test_benchmark_defaults(self):
        code, _, raw = self._get("/api/admin/benchmark", TOKEN)
        info = json.loads(raw)
        self.assertEqual(code, 200)
        self.assertEqual(info["default"]["k1"], server.retrieval.DEFAULT.k1)
        self.assertGreaterEqual(info["queries"]["traps"], 5)

    def test_benchmark_runs_in_runtime(self):
        code, report = self._post("/api/admin/benchmark", {"mode": "stem", "k1": 1.5, "b": 0.5}, TOKEN)
        self.assertEqual(code, 200)
        self.assertEqual(report["config"]["mode"], "stem")
        for key in ("recall@5", "recall@10", "mrr@10", "correct_rejection_rate"):
            self.assertIn(key, report["metrics"])
        self.assertNotIn("saved", report)

    def test_benchmark_rejects_bad_params(self):
        for payload, field in (({"mode": "magic"}, "mode"), ({"k1": "x"}, "k1"), ({"b": 2}, "b"),
                               ({"title_weight": 1.5}, "title_weight"), ({"expand": "yes"}, "expand")):
            code, body = self._post("/api/admin/benchmark", payload, TOKEN)
            self.assertEqual(code, 400, payload)
            self.assertIn(field, body["error"])

    def test_pages_served(self):
        code, _, html = self._get("/admin")
        self.assertEqual(code, 200)
        self.assertIn("Вход администратора", html.decode("utf-8"))
        self.assertIn('id="age"', self._get("/")[2].decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
