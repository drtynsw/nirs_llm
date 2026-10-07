"""Разбор актуальной редакции закона из текстового файла -> строки для data/raw/laws.jsonl.

Нужен на КТ2, чтобы заменить раннюю редакцию из открытого набора данных.
Как получить файл: открыть Трудовой кодекс на pravo.gov.ru или consultant.ru,
сохранить как текст (UTF-8). Статьи в файле начинаются со строк вида
«Статья 312.1. Общие положения».

Запуск:
    python scripts/parse_law_text.py tk_rf.txt --law "ТК РФ" --url https://... > data/raw/tk_current.jsonl
"""
import argparse
import json
import re
import sys

HEADER = re.compile(r"^\s*Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$")


def parse(text: str, law: str, law_title: str, url: str):
    article, title, buf = None, None, []
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            if article and buf:
                yield article, title, buf
            article, title, buf = m.group(1), m.group(2).strip(), []
        elif article and line.strip():
            buf.append(line.strip())
    if article and buf:
        yield article, title, buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--law", required=True, help="короткое имя: ТК РФ, ГК РФ, 422-ФЗ (самозанятые)")
    ap.add_argument("--law-title", default="")
    ap.add_argument("--url", default="")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read()
    n = 0
    for article, title, lines in parse(text, a.law, a.law_title, a.url):
        # утратившие силу статьи пропускаем
        if title.lower().startswith("утратила силу"):
            continue
        print(json.dumps({"law": a.law, "law_title": a.law_title, "source_url": a.url,
                          "article": article, "title": title, "text": "\n".join(lines)},
                         ensure_ascii=False))
        n += 1
    print(f"Статей: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
