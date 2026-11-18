"""HTTP-сервер приложения «Консультант по трудовому праву для студентов».

Только стандартная библиотека Python. Контракт API описан в spec/tech/api.md,
администрирование и аналитика — в spec/tech/admin.md.
"""
import hmac
import json
import time
from threading import Lock
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app import age as age_rules
from app import analytics, config, llm, llm_stub, rag, retrieval
from app.retrieval import preprocess
from benchmark import run as bench
from app.retrieval.corpus import load_corpus

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16 * 1024
MAX_QUESTION = 500
TOP_K_RANGE = (1, 10)
DAYS_RANGE = (1, 365)

config.load_dotenv()
ADMIN_TOKEN, ADMIN_TOKEN_GENERATED = config.admin_token()
STORE: analytics.Store | None = None   # создаётся при первом обращении; тесты подменяют


def get_store() -> analytics.Store:
    global STORE
    if STORE is None:
        STORE = analytics.Store(config.analytics_db())
    return STORE


def _with_note(answer: str, note: str | None) -> str:
    """Вставляет пояснение о возрасте перед оговоркой, чтобы оговорка оставалась последней."""
    if not note:
        return answer
    if answer.endswith(llm_stub.DISCLAIMER):
        return answer[: -len(llm_stub.DISCLAIMER)] + note + " " + llm_stub.DISCLAIMER
    return answer + " " + note


def ask(payload: dict) -> tuple[int, dict]:
    """Логика POST /api/ask, отделена от HTTP для тестов."""
    if not isinstance(payload, dict):
        return 400, {"error": "Некорректный JSON"}
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
    try:
        age = age_rules.validate(payload.get("age"))
    except ValueError as e:
        return 400, {"error": str(e)}

    t0 = time.perf_counter()
    result = rag.respond(question, top_k, age=age)
    docs = result["documents"]
    answer = _with_note(result["answer"], age_rules.note(age))
    age_docs = age_rules.documents(question, age, {d["doc_id"] for d in docs})
    took_ms = round((time.perf_counter() - t0) * 1000, 1)

    group = age_rules.group(age)
    try:
        get_store().record(question=question, age_group=group, top_k=top_k,
                           documents=docs, took_ms=took_ms)
    except Exception as e:  # аналитика не должна ломать ответ пользователю
        print(f"[analytics] не удалось записать событие: {e}")

    return 200, {
        "question": question,
        "answer": answer,
        "mode": result["mode"],       # rag | prototype | fallback
        "documents": docs,
        "unsupported_citations": result["unsupported_citations"],
        "age_group": group,
        "age_documents": age_docs,
        "took_ms": took_ms,
    }


def sources() -> dict:
    corpus = load_corpus()
    counts = Counter(d["law"] for d in corpus)
    return {"documents": len(corpus),
            "sources": [{"law": s, "documents": n} for s, n in counts.items()]}


def is_admin(authorization: str | None) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    given = authorization[len("Bearer "):].strip()
    return hmac.compare_digest(given.encode("utf-8"), ADMIN_TOKEN.encode("utf-8"))


def parse_days(query: dict) -> int:
    raw = query.get("days", ["30"])[0]
    days = int(raw)  # ValueError -> 400
    if not DAYS_RANGE[0] <= days <= DAYS_RANGE[1]:
        raise ValueError
    return days


def admin_stats(authorization: str | None, query: dict) -> tuple[int, dict]:
    """Логика GET /api/admin/stats, отделена от HTTP для тестов."""
    if not is_admin(authorization):
        return 401, {"error": "Нужен токен администратора"}
    try:
        days = parse_days(query)
    except ValueError:
        return 400, {"error": f"days должен быть целым числом от {DAYS_RANGE[0]} до {DAYS_RANGE[1]}"}
    return 200, get_store().stats(days)


_benchmark_lock = Lock()
BENCH_LIMITS = {"k1": (float, 0.1, 5.0), "b": (float, 0.0, 1.0),
                "title_weight": (int, 1, 5), "min_score": (float, 0.0, 100.0)}


def benchmark_config(payload: dict) -> retrieval.Config:
    """Конфигурация прогона из JSON; ValueError с именем неверного поля."""
    d = retrieval.DEFAULT
    values = {"mode": payload.get("mode", d.mode), "expand": payload.get("expand", d.expand)}
    if values["mode"] not in preprocess.MODES:
        raise ValueError("mode")
    if not isinstance(values["expand"], bool):
        raise ValueError("expand")
    for key, (cast, lo, hi) in BENCH_LIMITS.items():
        raw = payload.get(key, getattr(d, key))
        try:
            if isinstance(raw, bool) or (cast is int and isinstance(raw, float) and not raw.is_integer()):
                raise ValueError
            value = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(key) from None
        if not lo <= value <= hi:
            raise ValueError(key)
        values[key] = value
    return retrieval.Config(**values)


def admin_benchmark(authorization: str | None, payload: dict | None) -> tuple[int, dict]:
    """GET (payload=None) — параметры по умолчанию; POST — прогон benchmark в рантайме."""
    if not is_admin(authorization):
        return 401, {"error": "Нужен токен администратора"}
    if payload is None:
        queries = bench.load_queries()
        settings = llm.Settings.from_env()
        return 200, {"default": retrieval.DEFAULT.as_dict(),
                     "lemma_available": preprocess.lemma_available(),
                     "llm_enabled": settings.enabled, "llm": settings.describe(),
                     "modes": list(preprocess.MODES),
                     "queries": {"regular": sum(q["type"] == "regular" for q in queries),
                                 "traps": sum(q["type"] == "trap" for q in queries)}}
    if not isinstance(payload, dict):
        return 400, {"error": "Некорректный JSON"}
    try:
        config = benchmark_config(payload)
    except ValueError as e:
        return 400, {"error": f"Некорректный параметр: {e}"}
    settings = None
    if payload.get("llm", False) is True:
        settings = llm.Settings.from_env()
        if not settings.enabled:
            return 400, {"error": "LLM не настроена: задайте LLM_BASE_URL и LLM_MODEL в .env"}
    elif payload.get("llm", False) is not False:
        return 400, {"error": "Некорректный параметр: llm"}
    if not _benchmark_lock.acquire(blocking=False):
        return 409, {"error": "Уже выполняется benchmark"}
    try:
        report = bench.evaluate(config, settings=settings)
        if payload.get("save") is True:
            report["saved"] = bench.save(report).relative_to(bench.ROOT.parent).as_posix()
        return 200, report
    finally:
        _benchmark_lock.release()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict, headers: dict | None = None) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", headers)

    def do_GET(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        path, query = url.path, parse_qs(url.query)
        no_store = {"Cache-Control": "no-store"}
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path in ("/admin", "/admin/", "/admin.html"):
            self._send(200, (STATIC / "admin.html").read_bytes(), "text/html; charset=utf-8", no_store)
        elif path == "/api/health":
            self._json(200, {"status": "ok"})
        elif path == "/api/sources":
            self._json(200, sources())
        elif path == "/api/admin/stats":
            self._json(*admin_stats(self.headers.get("Authorization"), query), no_store)
        elif path == "/api/admin/benchmark":
            self._json(*admin_benchmark(self.headers.get("Authorization"), None), no_store)
        elif path == "/api/admin/export.csv":
            if not is_admin(self.headers.get("Authorization")):
                self._json(401, {"error": "Нужен токен администратора"}, no_store)
                return
            try:
                days = parse_days(query)
            except ValueError:
                self._json(400, {"error": "Некорректный days"}, no_store)
                return
            body = get_store().export_csv(days).encode("utf-8")
            self._send(200, body, "text/csv; charset=utf-8",
                       {**no_store, "Content-Disposition": f'attachment; filename="analytics_{days}d.csv"'})
        else:
            self._json(404, {"error": "Не найдено"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path not in ("/api/ask", "/api/admin/benchmark"):
            self._json(404, {"error": "Не найдено"})
            return
        admin = path == "/api/admin/benchmark"
        headers = {"Cache-Control": "no-store"} if admin else None
        if admin and not is_admin(self.headers.get("Authorization")):
            self._json(401, {"error": "Нужен токен администратора"}, headers)
            return
        lengths = self.headers.get_all("Content-Length", [])
        raw = lengths[0] if lengths else "0"
        if ("Transfer-Encoding" in self.headers or len(lengths) > 1
                or not raw.isascii() or not raw.isdecimal()):
            self._json(400, {"error": "Некорректная длина тела запроса"}, headers)
            return
        digits = raw.lstrip("0") or "0"
        if len(digits) > len(str(MAX_BODY_BYTES)) or int(digits) > MAX_BODY_BYTES:
            self._json(413, {"error": "Тело запроса превышает 16 КиБ"}, headers)
            return
        length = int(digits)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, RecursionError):
            self._json(400, {"error": "Некорректный JSON"}, headers)
            return
        if path == "/api/ask":
            self._json(*ask(payload))
        else:
            self._json(*admin_benchmark(self.headers.get("Authorization"), payload),
                       {"Cache-Control": "no-store"})

    def log_message(self, fmt, *args):  # короче стандартного лога
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    t0 = time.perf_counter()
    r = retrieval.get()   # индекс строим до первого запроса, чтобы он не ждал
    print(f"Индекс BM25: {r.index.N} документов, {len(r.index.postings)} термов, "
          f"нормализация {r.mode}, {time.perf_counter() - t0:.1f} с")
    print(f"Языковая модель: {llm.Settings.from_env().describe()}")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Консультант по трудовому праву: http://{host}:{port}  (Ctrl+C — остановить)")
    print(f"Страница администратора: http://{host}:{port}/admin")
    if ADMIN_TOKEN_GENERATED:
        print(f"Токен администратора (на этот запуск): {ADMIN_TOKEN}")
        print("Чтобы токен не менялся, задайте ADMIN_TOKEN в переменных окружения или в .env")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
