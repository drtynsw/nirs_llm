"""ВРЕМЕННЫЙ генератор ответа (контрольная точка 1).

Настоящий вызов LLM с промптом из spec/tech/rag.md подключается на КТ2
в app/llm.py с тем же интерфейсом: answer(question, docs) -> str.
"""

DISCLAIMER = "Это справка по тексту закона, а не юридическая консультация."


def answer(question: str, docs: list[dict]) -> str:
    if not docs:
        from app.rag import NO_ANSWER
        return NO_ANSWER
    refs = "; ".join(f"{d['citation']} «{d['title']}»" for d in docs[:3])
    return ("Прототип: здесь будет короткий ответ языковой модели, составленный только "
            f"по найденным статьям. Наиболее подходящие сейчас: {refs}. {DISCLAIMER}")
