"""Сборка корпуса: data/raw/laws.jsonl -> data/corpus.jsonl.

Документ корпуса = статья закона (spec/tech/rag.md, раздел «Документ»).
Длинные статьи (> MAX_WORDS слов) делятся на фрагменты по абзацам, чтобы
одна огромная статья (например, ст. 81 ТК РФ) не выигрывала поиск за счёт длины.

Запуск:  python scripts/build_corpus.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "laws.jsonl"
OUT = ROOT / "data" / "corpus.jsonl"

MAX_WORDS = 350     # длиннее — делим
CHUNK_WORDS = 250   # целевой размер фрагмента

LAW_CODE = {"ТК РФ": "tk", "ГК РФ": "gk", "422-ФЗ (самозанятые)": "npd"}


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # подпись президента и даты в конце закона — не содержание статьи
    text = re.split(r"\n\s*Президент\s+Российской Федерации", text)[0]
    return text.strip()


def split_long(text: str) -> list[str]:
    if len(text.split()) <= MAX_WORDS:
        return [text]
    chunks, cur, n = [], [], 0
    for para in text.split("\n"):
        w = len(para.split())
        if cur and n + w > CHUNK_WORDS:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def main() -> None:
    docs = []
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            parts = split_long(clean(art["text"]))
            for i, part in enumerate(parts, start=1):
                suffix = f"_{i}" if len(parts) > 1 else ""
                docs.append({
                    "doc_id": f"{LAW_CODE[art['law']]}_{art['article']}{suffix}",
                    "law": art["law"],
                    "article": art["article"],
                    "fragment": i if len(parts) > 1 else None,
                    "title": art["title"],
                    "citation": f"ст. {art['article']} {art['law'].split(' (')[0]}"
                                + (f", фрагмент {i}" if len(parts) > 1 else ""),
                    "text": part,
                })
    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Документов: {len(docs)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
