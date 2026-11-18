"""Конвейер ответа: поиск BM25 -> порог «нет ответа» (С4) -> ответ LLM.

Один и тот же код вызывают сервер (POST /api/ask) и benchmark, поэтому
benchmark проверяет ровно то, что видит пользователь.
"""
from app import llm, llm_stub, retrieval

NO_ANSWER = ("В загруженных законах не нашлось подходящих статей. "
             "Попробуйте описать ситуацию другими словами.")
DISCLAIMER = llm_stub.DISCLAIMER
DISABLED = llm.Settings(base_url="", model="")


def _refusal(mode: str) -> dict:
    # С4: статьи не показываем — они не отвечают на вопрос.
    return {"documents": [], "answer": NO_ANSWER, "refused": True, "mode": mode,
            "unsupported_citations": [], "llm_error": None}


def respond(question: str, top_k: int, config: retrieval.Config = retrieval.DEFAULT,
            age: int | None = None, settings: llm.Settings | None = None) -> dict:
    """{"documents", "answer", "refused", "mode", "unsupported_citations", "llm_error"}

    mode: "rag" — ответила LLM; "prototype" — LLM не настроена; "fallback" — LLM упала.
    """
    settings = settings if settings is not None else llm.Settings.from_env()
    r = retrieval.get(config)
    docs = r.search(question, top_k)
    mode = "rag" if settings.enabled else "prototype"
    if not docs or not r.answerable(docs[0]["score"]):
        return _refusal(mode)

    if settings.enabled:
        context = [{**d, "text": r.corpus[d["doc_id"]]["text"]} for d in docs]
        try:
            out = llm.answer(question, context, age, settings)
        except llm.LLMError as e:
            print(f"[llm] {e}")
            return {"documents": docs, "answer": llm_stub.answer(question, docs), "refused": False,
                    "mode": "fallback", "unsupported_citations": [], "llm_error": str(e)}
        if out["refused"]:
            return _refusal("rag")
        return {"documents": docs, "answer": f"{out['text']} {DISCLAIMER}", "refused": False,
                "mode": "rag", "unsupported_citations": out["unsupported"], "llm_error": None}

    return {"documents": docs, "answer": llm_stub.answer(question, docs), "refused": False,
            "mode": "prototype", "unsupported_citations": [], "llm_error": None}
