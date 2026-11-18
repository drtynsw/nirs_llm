"""Подключение LLM на имитаторе OpenAI-совместимого API. Запуск: python -m unittest -v"""
import json
import os
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from app import analytics, llm, rag, server
from benchmark import run as bench

server.STORE = analytics.Store(":memory:")
llm.RETRY_DELAY = 0   # повторы без пауз, чтобы тесты были быстрыми


class FakeLLM(BaseHTTPRequestHandler):
    """Отвечает как chat/completions. Поведение задаётся FakeLLM.behavior."""
    behavior = "cite"      # cite | refuse | error | bad | slow | foreign | flaky | empty
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        user = body["messages"][-1]["content"]
        if FakeLLM.behavior == "error":
            self._reply(500, b'{"error": "boom"}')
            return
        if FakeLLM.behavior == "empty" or (FakeLLM.behavior == "flaky" and len(FakeLLM.requests) == 1):
            self._reply(200, json.dumps({"choices": [{"message": {"content": ""}}]}).encode())
            return
        if FakeLLM.behavior == "bad":
            self._reply(200, b"not json")
            return
        if FakeLLM.behavior == "slow":
            time.sleep(1.5)
        if FakeLLM.behavior == "refuse" or "стипенд" in user or "вычет за обучение" in user:
            text = "НЕТ_ОТВЕТА"
        elif FakeLLM.behavior == "foreign":
            text = "Так нельзя по ст. 999 ТК РФ. Это не является юридической консультацией."
        else:
            first = re.search(r"\[(ст\. [\d.]+ [^\],]+)", user).group(1)
            text = f"Коротко: смотрите {first}, там всё сказано. Это не юридическая консультация."
        data = {"choices": [{"message": {"role": "assistant", "content": text}}]}
        self._reply(200, json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _reply(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class LLMTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.settings = llm.Settings(base_url=f"http://127.0.0.1:{cls.httpd.server_address[1]}/v1",
                                    model="fake-model", api_key="test-key", auth_scheme="Api-Key",
                                    project="folder-1", timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        FakeLLM.behavior = "cite"
        FakeLLM.requests = []


class PromptTest(unittest.TestCase):
    DOCS = [{"citation": "ст. 70 ТК РФ", "title": "Испытание при приеме на работу", "article": "70",
             "text": "Срок испытания не может превышать трех месяцев."}]

    def test_messages_have_instruction_question_and_context(self):
        msgs = llm.build_messages("Сколько длится испытательный срок?", self.DOCS)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("НЕТ_ОТВЕТА", msgs[0]["content"])
        self.assertIn("Сколько длится испытательный срок?", msgs[1]["content"])
        self.assertIn("[ст. 70 ТК РФ] Испытание при приеме на работу", msgs[1]["content"])
        self.assertIn("трех месяцев", msgs[1]["content"])

    def test_minor_age_mentioned_only_for_minors(self):
        self.assertIn("16 лет", llm.build_messages("вопрос", self.DOCS, age=16)[1]["content"])
        self.assertNotIn("лет:", llm.build_messages("вопрос", self.DOCS, age=20)[1]["content"])

    def test_citations_and_unsupported(self):
        text = "По ст. 70 ТК РФ и статье 71, а также статьями 173 и ст. 312.1 ТК РФ."
        self.assertEqual(llm.cited_articles(text), ["70", "71", "173", "312.1"])
        self.assertEqual(llm.unsupported(text, self.DOCS), ["71", "173", "312.1"])

    def test_citation_matches_law_and_context(self):
        docs = [{"citation": "ст. 4 ТК РФ"}, {"citation": "ст. 4 422-ФЗ"}]
        self.assertEqual(llm.unsupported("ст. 4 ГК РФ", docs), ["4"])
        self.assertEqual(llm.unsupported("ст. 4", docs), ["4"])
        self.assertEqual(llm.unsupported("ст. 4", docs[:1]), [])
        self.assertEqual(llm.unsupported("ст. 4 422-ФЗ", docs), [])
        self.assertEqual(llm.unsupported("ст. 4 422-ФЗ", docs[:1] * 5 + docs[1:]), ["4"])

    def test_accuracy_requires_relevant_law(self):
        row = {"mode": "rag", "refused": False, "unsupported_citations": [],
               "top": [{"citation": "ст. 4 422-ФЗ"}], "answer": "См. ст. 4 422-ФЗ."}
        self.assertFalse(bench.answer_correct(row, ["tk_4"]))

    def test_model_disclaimer_removed(self):
        self.assertEqual(llm.clean("Можно (ст. 80 ТК РФ). Это не является юридической консультацией."),
                         "Можно (ст. 80 ТК РФ).")

    def test_truncated_answer_trimmed_to_last_sentence(self):
        text = "Отпуск — 28 дней (ст. 115 ТК РФ). По ч. 2 ст. 4 нельзя. Если же вы учитесь очно,"
        self.assertEqual(llm.trim_incomplete(text), "Отпуск — 28 дней (ст. 115 ТК РФ). По ч. 2 ст. 4 нельзя.")

    def test_markdown_removed(self):
        self.assertEqual(llm.clean("Да, **с сохранением** заработка (ст. 173 ТК РФ)."),
                         "Да, с сохранением заработка (ст. 173 ТК РФ).")

    def test_refusal_marker(self):
        self.assertTrue(llm.is_refusal("НЕТ_ОТВЕТА"))
        self.assertTrue(llm.is_refusal("нет ответа"))
        self.assertTrue(llm.is_refusal("«НЕТ_ОТВЕТА»."))
        self.assertFalse(llm.is_refusal("Ответ: можно."))
        self.assertFalse(llm.is_refusal("Про стипендию в статьях нет ответа, но про отпуск — ст. 173 ТК РФ."))

    def test_disabled_without_url_or_model(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "", "LLM_MODEL": ""}):
            self.assertFalse(llm.Settings.from_env().enabled)


class RespondWithLLMTest(LLMTestBase):
    Q = "максимальный срок испытания при приеме на работу"

    def test_request_format_and_headers(self):
        rag.respond(self.Q, 5, settings=self.settings)
        req = FakeLLM.requests[-1]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Api-Key test-key")
        self.assertEqual(req["headers"]["openai-project"], "folder-1")
        self.assertEqual(req["body"]["model"], "fake-model")
        # в контексте — полный текст статьи, а не фрагмент из выдачи
        self.assertIn("Испытание при приеме на работу", req["body"]["messages"][1]["content"])

    def test_answer_mode_rag_with_single_disclaimer(self):
        out = rag.respond(self.Q, 5, settings=self.settings)
        self.assertEqual(out["mode"], "rag")
        self.assertTrue(out["documents"])
        self.assertTrue(out["answer"].endswith(rag.DISCLAIMER))
        self.assertEqual(out["answer"].count("юридическ"), 1)
        self.assertEqual(out["unsupported_citations"], [])

    def test_model_refusal_becomes_c4(self):
        FakeLLM.behavior = "refuse"
        out = rag.respond(self.Q, 5, settings=self.settings)
        self.assertTrue(out["refused"])
        self.assertEqual(out["documents"], [])
        self.assertEqual(out["answer"], rag.NO_ANSWER)

    def test_unsupported_citation_reported(self):
        FakeLLM.behavior = "foreign"
        self.assertEqual(rag.respond(self.Q, 5, settings=self.settings)["unsupported_citations"], ["999"])

    def test_below_threshold_model_not_called(self):
        out = rag.respond("какая погода будет в Москве на выходных", 5, settings=self.settings)
        self.assertTrue(out["refused"])
        self.assertEqual(FakeLLM.requests, [])

    def test_errors_fall_back_to_stub_with_documents(self):
        for behavior in ("error", "bad"):
            FakeLLM.behavior = behavior
            out = rag.respond(self.Q, 5, settings=self.settings)
            self.assertEqual(out["mode"], "fallback", behavior)
            self.assertTrue(out["documents"])
            self.assertTrue(out["llm_error"])

    def test_provider_error_details_are_private(self):
        FakeLLM.behavior = "error"
        with mock.patch("builtins.print") as logged:
            result = rag.respond(self.Q, 5, settings=self.settings)
        self.assertEqual(result["llm_error"], "HTTP 500")
        logged.assert_called_once_with("[llm] HTTP 500")

    def test_transport_error_details_are_private(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("private details")):
            with self.assertRaisesRegex(llm.LLMError, "^нет связи с моделью$"):
                llm.complete([], self.settings)

    def test_transient_empty_answer_retried(self):
        FakeLLM.behavior = "flaky"
        out = rag.respond(self.Q, 5, settings=self.settings)
        self.assertEqual(out["mode"], "rag")
        self.assertEqual(len(FakeLLM.requests), 2)

    def test_retries_are_limited(self):
        FakeLLM.behavior = "empty"
        self.assertEqual(rag.respond(self.Q, 5, settings=self.settings)["mode"], "fallback")
        self.assertEqual(len(FakeLLM.requests), 1 + llm.RETRIES)

    def test_server_error_retried(self):
        FakeLLM.behavior = "error"   # HTTP 500 — временный сбой, повторяем
        rag.respond(self.Q, 5, settings=self.settings)
        self.assertEqual(len(FakeLLM.requests), 1 + llm.RETRIES)

    def test_timeout_falls_back(self):
        FakeLLM.behavior = "slow"
        fast = llm.Settings(**{**self.settings.__dict__, "timeout": 0.3})
        self.assertEqual(rag.respond(self.Q, 5, settings=fast)["mode"], "fallback")

    def test_server_uses_env_settings(self):
        env = {"LLM_BASE_URL": self.settings.base_url, "LLM_MODEL": "fake-model", "LLM_API_KEY": ""}
        with mock.patch.dict(os.environ, env):
            code, body = server.ask({"question": self.Q, "age": 16})
        self.assertEqual(code, 200)
        self.assertEqual(body["mode"], "rag")
        self.assertIn("Вам 16 лет", body["answer"])
        self.assertTrue(body["answer"].endswith(rag.DISCLAIMER))
        self.assertNotIn("authorization", FakeLLM.requests[-1]["headers"])

    def test_without_llm_mode_is_prototype(self):
        out = rag.respond(self.Q, 5, settings=rag.DISABLED)
        self.assertEqual(out["mode"], "prototype")


class BenchmarkWithLLMTest(LLMTestBase):
    def test_accuracy_and_rejections_with_llm(self):
        report = bench.evaluate(settings=self.settings)
        m = report["metrics"]
        self.assertIsNotNone(m["accuracy"])
        self.assertGreater(m["accuracy"], 0)
        self.assertEqual(m["llm_errors"], 0)
        # имитатор отказывается на «стипендию» и «вычет за обучение» -> все 6 ловушек отклонены
        self.assertEqual(m["correct_rejection_rate"], 1.0)
        self.assertIn("fake-model", report["llm"])

    def test_without_llm_accuracy_is_none(self):
        self.assertIsNone(bench.evaluate()["metrics"]["accuracy"])
        self.assertEqual(FakeLLM.requests, [])


if __name__ == "__main__":
    unittest.main()
