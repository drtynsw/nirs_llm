"""HTTP-сервер приложения «Консультант по трудовому праву для студентов».

Только стандартная библиотека Python. Контракт API описан в spec/tech/api.md.
"""
import json
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import llm_stub, search_stub

STATIC = Path(__file__).resolve().parent / "static"
MAX_QUESTION = 500
TOP_K_RANGE = (1, 10)


def ask(payload: dict) -> tuple[int, dict]:
    """Логика POST /api/ask, отделена от HTTP для тестов."""
    question = str(payload.get("question", "")).strip()
    if not question:
        return 400, {"error": "Пустой вопрос"}
    if len(question) > MAX_QUESTION:
        return 400, {"error": f"Вопрос длиннее {MAX_QUESTION} символов"}
    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError):
        return 400, {"error": "top_k должен быть числом"}
    top_k = max(TOP_K_RANGE[0], min(TOP_K_RANGE[1], top_k))

    t0 = time.perf_counter()
    docs = search_stub.search(question, top_k)
    answer = llm_stub.answer(question, docs)
    return 200, {
        "question": question,
        "answer": answer,
        "mode": "prototype",          # на КТ2 станет "rag"
        "documents": docs,
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def sources() -> dict:
    corpus = search_stub.load_corpus()
    counts = Counter(d["law"] for d in corpus)
    return {"documents": len(corpus),
            "sources": [{"law": s, "documents": n} for s, n in counts.items()]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/api/sources":
            self._json(200, sources())
        else:
            self._json(404, {"error": "Не найдено"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self._json(404, {"error": "Не найдено"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Некорректный JSON"})
            return
        self._json(*ask(payload))

    def log_message(self, fmt, *args):  # короче стандартного лога
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Консультант по трудовому праву: http://{host}:{port}  (Ctrl+C — остановить)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
