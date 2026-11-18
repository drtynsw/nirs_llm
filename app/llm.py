"""Генерация ответа языковой моделью по найденным статьям (spec/tech/rag.md).

Клиент OpenAI-совместимого API (POST {LLM_BASE_URL}/chat/completions) на
стандартной библиотеке: YandexGPT, Ollama, DeepSeek, OpenRouter и др.
Настройки — только переменные окружения / .env (AGENTS.md).
"""
import json
import os
import time
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

REFUSAL_MARKER = "НЕТ_ОТВЕТА"
MAX_TOKENS = 1500   # с запасом: у моделей с рассуждениями они входят в этот лимит
MAX_DOCS = 5
MAX_DOC_CHARS = 1500

SYSTEM_PROMPT = f"""Ты — справочный помощник по трудовому праву для школьников и студентов 14–23 лет, \
которые подрабатывают. Отвечай только на основе статей законов из контекста.

Правила:
1. Пиши по-русски, простым языком, на «вы», 2–4 предложения.
2. Каждое утверждение подкрепляй ссылкой на статью из контекста в форме «ст. 70 ТК РФ», \
«ст. 779 ГК РФ», «ст. 10 422-ФЗ». Не ссылайся на статьи, которых нет в контексте.
3. Юридические термины поясняй обычными словами.
4. Не давай советов сверх текста закона, не составляй документы, не выдумывай цифры.
5. Если статьи отвечают на вопрос хотя бы частично — ответь тем, что в них есть, и коротко скажи, чего в них нет.
6. Только если статьи совсем не о том, о чём спрашивают, ответь ровно одним словом: {REFUSAL_MARKER}
7. Не добавляй оговорку про юридическую консультацию — её добавит система.
8. Пиши обычным текстом, без Markdown: без звёздочек, решёток и списков."""


class LLMError(Exception):
    """Модель недоступна или вернула некорректный ответ."""


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str = ""
    auth_scheme: str = "Bearer"
    project: str = ""
    timeout: float = 30.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "").strip().rstrip("/"),
            model=os.environ.get("LLM_MODEL", "").strip(),
            api_key=os.environ.get("LLM_API_KEY", "").strip(),
            auth_scheme=os.environ.get("LLM_AUTH_SCHEME", "Bearer").strip() or "Bearer",
            project=os.environ.get("LLM_PROJECT", "").strip(),
            timeout=float(os.environ.get("LLM_TIMEOUT", "30") or 30),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0") or 0),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}" if self.enabled else "не настроена (заглушка)"


def build_messages(question: str, docs: list[dict], age: int | None = None) -> list[dict]:
    """Запрос к модели: инструкция + вопрос + контекст с идентификаторами статей."""
    context = "\n\n".join(
        f"[{d['citation']}] {d['title']}\n{' '.join(d.get('text', d.get('snippet', '')).split())[:MAX_DOC_CHARS]}"
        for d in docs[:MAX_DOCS])
    about = ""
    if age is not None and age < 18:
        about = (f"\nПользователю {age} лет: если в контексте есть нормы для работников младше "
                 "18 лет, опирайся на них.")
    user = f"Вопрос: {question}{about}\n\nКонтекст — статьи законов:\n\n{context}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class _Transient(LLMError):
    """Сбой, который имеет смысл повторить: сеть, 429/5xx, пустой ответ."""


RETRIES = 2          # всего попыток = 1 + RETRIES
RETRY_DELAY = 1.0    # с, растёт линейно


def complete(messages: list[dict], settings: Settings) -> str:
    """chat/completions с повтором временных сбоев. LLMError, если ответа так и нет."""
    for attempt in range(RETRIES + 1):
        try:
            return _complete_once(messages, settings)
        except _Transient as e:
            if attempt == RETRIES:
                raise LLMError(str(e)) from None
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise AssertionError("unreachable")


def _complete_once(messages: list[dict], settings: Settings) -> str:
    body = json.dumps({"model": settings.model, "messages": messages,
                       "temperature": settings.temperature, "max_tokens": MAX_TOKENS}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"{settings.auth_scheme} {settings.api_key}"
    if settings.project:
        headers["OpenAI-Project"] = settings.project
    req = urllib.request.Request(settings.base_url + "/chat/completions", body, headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=settings.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        e.close()
        error = _Transient if e.code == 429 or e.code >= 500 else LLMError
        raise error(f"HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise _Transient("нет связи с моделью") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise LLMError("ответ модели — не JSON") from None
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("в ответе модели нет choices[0].message.content") from None
    if not isinstance(content, str) or not content.strip():
        raise _Transient("модель вернула пустой ответ")
    if choice.get("finish_reason") == "length":
        print("[llm] ответ обрезан лимитом max_tokens — отрезаю незаконченное предложение")
        content = trim_incomplete(content)
    return content.strip()


# Конец предложения: . ! ? (возможно, после скобки или кавычки), но не сокращения «ст.», «ч.», «п.».
_SENTENCE_END = re.compile(r"(?<!ст)(?<!ч)(?<!п)(?<!гл)[.!?][)»\"]?(?=\s|$)")


def trim_incomplete(text: str) -> str:
    """Обрезает незаконченное последнее предложение (ответ упёрся в лимит токенов)."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[: ends[-1].end()] if ends else text


def is_refusal(text: str) -> bool:
    """Отказ — ответ, который начинается с маркера (а не упоминание «нет ответа» внутри текста)."""
    head = text.strip().strip("«»\"'*`.").upper().replace(" ", "_")
    return head.startswith(REFUSAL_MARKER)


_CITE = re.compile(
    r"(?:ст\.|стать[яиеюейм]+)\s*(\d+(?:\.\d+)?)"
    r"(?:\s+(?:№\s*)?([а-яё]{1,6}\s+РФ|\d+\s*[-‑–]\s*ФЗ))?", re.IGNORECASE)
_DISCLAIMER_TAIL = re.compile(r"\s*\(?[^.]*не (?:является )?юридическ\w* консультац\w*[^.]*\.?\)?\s*$",
                              re.IGNORECASE)


def citations(text: str) -> list[tuple[str, str]]:
    """Пары (номер, закон); пустая строка означает ссылку без названия закона."""
    return [(number, re.sub(r"\s+", "", law).upper().replace("‑", "-").replace("–", "-"))
            for number, law in _CITE.findall(text)]


def cited_articles(text: str) -> list[str]:
    return list(dict.fromkeys(number for number, _ in citations(text)))


def check_citations(text: str, docs: list[dict]) -> tuple[set[tuple[str, str]], list[str]]:
    allowed = {ref for doc in docs for ref in citations(doc["citation"])}
    supported, missing = set(), []
    for number, law in citations(text):
        matches = {ref for ref in allowed if ref[0] == number and (not law or ref[1] == law)}
        if len(matches) == 1:
            supported.update(matches)
        elif number not in missing:
            missing.append(number)
    return supported, missing


def unsupported(text: str, docs: list[dict]) -> list[str]:
    return check_citations(text, docs[:MAX_DOCS])[1]


_MARKDOWN = re.compile(r"\*\*|__|^#+\s*|^\s*[-*]\s+", re.MULTILINE)


def clean(text: str) -> str:
    """Убирает разметку Markdown, собственную оговорку модели (сервер добавит свою) и лишние пробелы."""
    text = _MARKDOWN.sub("", text)
    return " ".join(_DISCLAIMER_TAIL.sub("", text).split())


def answer(question: str, docs: list[dict], age: int | None = None,
           settings: Settings | None = None) -> dict:
    """{"text": str, "refused": bool, "unsupported": [...]}; LLMError, если модель недоступна."""
    settings = settings or Settings.from_env()
    raw = complete(build_messages(question, docs, age), settings)
    if is_refusal(raw):
        return {"text": "", "refused": True, "unsupported": []}
    text = clean(raw)
    return {"text": text, "refused": False, "unsupported": unsupported(text, docs)}
