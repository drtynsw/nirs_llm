"""Учёт возраста пользователя (КТ2, сценарий С6 в spec/product/scenarios.md).

Для работников младше 18 лет Трудовой кодекс устанавливает особые правила
(ст. 63, 92, 94, 96, гл. 42). Модуль не меняет ранжирование поиска: он
добавляет пояснение к ответу и отдельный список статей `age_documents`.
"""
import re

from app.retrieval.corpus import load_corpus

MIN_AGE, MAX_AGE = 10, 100
GROUPS = ("до 14", "14–15", "16–17", "18–23", "24+")
AGE_ERROR = f"Возраст должен быть целым числом от {MIN_AGE} до {MAX_AGE}"

# Тема вопроса (корни слов) -> статьи ТК РФ для работников младше 18 лет.
_TOPIC_ARTICLES = [
    (r"ноч|выходн|праздн|сверхуроч|переработ|командиров", ["tk_268", "tk_96"]),
    (r"час|смен|график|врем|недел|учеб|учёб|школ", ["tk_92", "tk_94"]),
    (r"отпуск|каникул|лет[оа]м?\b", ["tk_267"]),
    (r"увол|расторж|выгна|сокращ", ["tk_269"]),
    (r"зарплат|заработн|плат|деньг|норм[аы] выработ", ["tk_271", "tk_270"]),
    (r"вредн|опасн|тяж[её]л|груз|склад|бар|алкогол|табак", ["tk_265"]),
    (r"устро|оформ|договор|прин[яи]|возраст|лет\b|медосмотр|справк|родител", ["tk_63", "tk_266"]),
]
_DEFAULT_ARTICLES = ["tk_63", "tk_268", "tk_92"]
MAX_AGE_DOCS = 3


def validate(raw) -> int | None:
    """None, если возраст не передан; иначе целое 10–100. ValueError с текстом для 400."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ValueError(AGE_ERROR)
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(AGE_ERROR)
        raw = int(raw)
    try:
        age = int(raw)
    except (TypeError, ValueError):
        raise ValueError(AGE_ERROR) from None
    if not MIN_AGE <= age <= MAX_AGE:
        raise ValueError(AGE_ERROR)
    return age


def group(age: int | None) -> str | None:
    if age is None:
        return None
    if age < 14:
        return "до 14"
    if age < 16:
        return "14–15"
    if age < 18:
        return "16–17"
    if age <= 23:
        return "18–23"
    return "24+"


def _years(age: int) -> str:
    if age % 10 == 1 and age % 100 != 11:
        return f"{age} год"
    if age % 10 in (2, 3, 4) and age % 100 not in (12, 13, 14):
        return f"{age} года"
    return f"{age} лет"


def note(age: int | None) -> str | None:
    """Пояснение к ответу для несовершеннолетних; для 18+ и без возраста — None."""
    if age is None or age >= 18:
        return None
    common = ("Для работников младше 18 лет рабочее время сокращено (ст. 92, 94 ТК РФ), "
              "нельзя привлекать к ночной и сверхурочной работе, работе в выходные и "
              "командировкам (ст. 268 ТК РФ), отпуск — 31 календарный день в удобное время "
              "(ст. 267 ТК РФ), при приёме и ежегодно — бесплатный медосмотр (ст. 266 ТК РФ). "
              "Уволить по инициативе работодателя можно только с согласия трудовой инспекции "
              "и комиссии по делам несовершеннолетних (ст. 269 ТК РФ).")
    if age < 14:
        return (f"Вам {_years(age)}: по общему правилу трудовой договор заключают с 16 лет. "
                "Младше 14 лет — только в кино, театре, концертных организациях и цирке "
                "с согласия родителя и органа опеки (ст. 63 ТК РФ). " + common)
    if age < 16:
        return (f"Вам {_years(age)}: трудовой договор обычно заключают с 16 лет; с 15 лет — "
                "после получения основного общего образования, с 14 лет — только лёгкий труд "
                "в свободное от учёбы время и с согласия родителя и органа опеки "
                "(ст. 63 ТК РФ). " + common)
    return f"Вам {_years(age)}: до 18 лет действуют особые правила. " + common


def documents(question: str, age: int | None, exclude: set[str] = frozenset()) -> list[dict]:
    """1–3 статьи о труде несовершеннолетних по теме вопроса, без повторов из exclude."""
    if age is None or age >= 18:
        return []
    q = question.lower()
    picked: list[str] = ["tk_63"] if age < 16 else []
    for pattern, ids in _TOPIC_ARTICLES:
        if re.search(pattern, q):
            picked += ids
    picked += _DEFAULT_ARTICLES
    by_id = {d["doc_id"]: d for d in load_corpus()}
    out, seen = [], set(exclude)
    for doc_id in picked:
        if doc_id in seen or doc_id not in by_id:
            continue
        seen.add(doc_id)
        d = by_id[doc_id]
        text = " ".join(d["text"].split())
        out.append({
            "doc_id": d["doc_id"], "citation": d["citation"], "law": d["law"],
            "article": d["article"], "title": d["title"],
            "snippet": text[:280] + ("…" if len(text) > 280 else ""),
        })
        if len(out) == MAX_AGE_DOCS:
            break
    return out
