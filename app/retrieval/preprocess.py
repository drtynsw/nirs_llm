"""Предобработка текста — одинаковая для документов и запросов (spec/tech/rag.md).

Шаги: нижний регистр, ё → е, токенизация регулярным выражением, стоп-слова,
нормализация одним из способов:
    none  — слово как есть;
    stem  — стеммер Snowball (app/retrieval/stemmer.py);
    lemma — лемматизация pymorphy3 (если библиотека не установлена — stem).
"""
import re
from functools import lru_cache

from app.retrieval.stemmer import stem

TOKEN = re.compile(r"[а-яёa-z0-9]+(?:\.[0-9]+)?")
MODES = ("none", "stem", "lemma")

# Служебные слова и слова-шаблоны вопросов. Проверяются и в исходной, и в
# нормализованной форме, поэтому достаточно начальных форм и частых словоформ.
STOPWORDS = frozenset("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до
его ее ей ему если есть еще же за здесь и из или им их к как ко когда кто ли либо мне меня мной
может мой мы на над надо наш не него нее нет ни них но ну о об однако он она они оно от очень по
под при про с со так также такой там те тем то того тоже той только том ты у уже хотя чего чей
чем что чтобы чье чья эта эти это этого этой этом этот я
можно ли какой какая какое какие каким сколько почему зачем нужно нужен надо мне меня мой моя
мои свой своя свои себя себе ваш ваша это этот всё весь сам самый вообще просто ещё уже
""".split())


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3
    except ImportError:
        return None
    return pymorphy3.MorphAnalyzer()


def lemma_available() -> bool:
    return _morph() is not None


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    morph = _morph()
    if morph is None:
        return stem(word)
    return morph.parse(word)[0].normal_form.replace("ё", "е")


def resolve_mode(mode: str) -> str:
    """Фактический способ нормализации: lemma без pymorphy3 превращается в stem."""
    if mode not in MODES:
        raise ValueError(f"Неизвестная нормализация: {mode}")
    if mode == "lemma" and not lemma_available():
        return "stem"
    return mode


def normalize(word: str, mode: str) -> str:
    if mode == "lemma":
        return _lemma(word)
    if mode == "stem":
        return stem(word)
    return word


def raw_tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower().replace("ё", "е"))


def terms(text: str, mode: str) -> list[str]:
    """Термы текста: токены без стоп-слов после нормализации."""
    out = []
    for tok in raw_tokens(text):
        if tok in STOPWORDS:
            continue
        norm = normalize(tok, mode)
        if norm and norm not in STOPWORDS:
            out.append(norm)
    return out
