"""Стеммер Портера (Snowball) для русского языка.

Собственная реализация алгоритма https://snowballstem.org/algorithms/russian/stemmer.html.
Нужен как вариант нормализации для сравнения на benchmark и как запасной
вариант, если pymorphy3 не установлен (приложение запускается без зависимостей).
"""
import re
from functools import lru_cache

_VOWELS = "аеиоуыэюя"


def _ends(word: str, start: int, groups: list[tuple[tuple[str, ...], bool]]) -> str | None:
    """Самое длинное окончание из групп внутри word[start:].

    Для групп с флагом True окончание должно стоять после «а» или «я»
    (сама буква остаётся в основе). Возвращает слово без окончания или None.
    """
    best = None
    for endings, after_a in groups:
        for e in endings:
            if not word.endswith(e) or len(word) - len(e) < start:
                continue
            if after_a:
                pos = len(word) - len(e) - 1
                if pos < start or word[pos] not in "ая":
                    continue
            if best is None or len(e) > len(best):
                best = e
    return word[: len(word) - len(best)] if best is not None else None


_PERFECTIVE = [(("в", "вши", "вшись"), True), (("ив", "ивши", "ившись", "ыв", "ывши", "ывшись"), False)]
_REFLEXIVE = [(("ся", "сь"), False)]
_ADJECTIVE = ("ее", "ие", "ые", "ое", "ими", "ыми", "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом",
              "его", "ого", "ему", "ому", "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею")
_PARTICIPLE = [(("ем", "нн", "вш", "ющ", "щ"), True), (("ивш", "ывш", "ующ"), False)]
_VERB = [(("ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло", "но", "ет", "ют", "ны", "ть",
           "ешь", "нно"), True),
         (("ила", "ыла", "ена", "ейте", "уйте", "ите", "или", "ыли", "ей", "уй", "ил", "ыл", "им",
           "ым", "ен", "ило", "ыло", "ено", "ят", "ует", "уют", "ит", "ыт", "ены", "ить", "ыть",
           "ишь", "ую", "ю"), False)]
_NOUN = [(("а", "ев", "ов", "ие", "ье", "е", "иями", "ями", "ами", "еи", "ии", "и", "ией", "ей", "ой",
           "ий", "й", "иям", "ям", "ием", "ем", "ам", "ом", "о", "у", "ах", "иях", "ях", "ы", "ь",
           "ию", "ью", "ю", "ия", "ья", "я"), False)]
_SUPERLATIVE = [(("ейш", "ейше"), False)]
_DERIVATIONAL = [(("ост", "ость"), False)]


def _regions(word: str) -> tuple[int, int]:
    """Начала областей RV и R2."""
    rv = next((i + 1 for i, c in enumerate(word) if c in _VOWELS), len(word))
    r1 = next((i + 1 for i in range(1, len(word)) if word[i] not in _VOWELS and word[i - 1] in _VOWELS),
              len(word))
    r2 = next((i + 1 for i in range(r1 + 1, len(word)) if word[i] not in _VOWELS and word[i - 1] in _VOWELS),
              len(word))
    return rv, r2


def _adjectival(word: str, rv: int) -> str | None:
    base = _ends(word, rv, [(_ADJECTIVE, False)])
    if base is None:
        return None
    return _ends(base, rv, _PARTICIPLE) or base


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    if not re.fullmatch(r"[а-я]+", word):
        return word
    rv, r2 = _regions(word)

    # Шаг 1
    base = _ends(word, rv, _PERFECTIVE)
    if base is not None:
        word = base
    else:
        word = _ends(word, rv, _REFLEXIVE) or word
        for step in (lambda w: _adjectival(w, rv), lambda w: _ends(w, rv, _VERB),
                     lambda w: _ends(w, rv, _NOUN)):
            base = step(word)
            if base is not None:
                word = base
                break
    # Шаг 2
    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]
    # Шаг 3
    word = _ends(word, r2, _DERIVATIONAL) or word
    # Шаг 4
    if word.endswith("нн") and len(word) - 1 >= rv:
        word = word[:-1]
    else:
        base = _ends(word, rv, _SUPERLATIVE)
        if base is not None:
            word = base[:-1] if base.endswith("нн") else base
        elif word.endswith("ь") and len(word) - 1 >= rv:
            word = word[:-1]
    return word
