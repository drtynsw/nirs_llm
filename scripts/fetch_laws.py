"""Выгрузка текстов законов из открытого набора данных -> data/raw/laws.jsonl.

Источник: npm-пакет @ansvar/russian-law-mcp 0.1.0 (Apache-2.0), внутри SQLite-база
с текстами федеральных законов, собранными с pravo.gov.ru. Сами тексты законов
объектом авторского права не являются (п. 6 ст. 1259 ГК РФ).

ВАЖНО: в этом наборе ранние редакции законов (нет, например, ст. 312.1 ТК РФ
о дистанционной работе и ст. 66.1 об электронной трудовой книжке). Для КТ1 этого
хватает, на КТ2 заменяем на актуальную редакцию: scripts/parse_law_text.py.

Запуск (нужен npm):  python scripts/fetch_laws.py
"""
import json
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "laws.jsonl"
PACKAGE = "@ansvar/russian-law-mcp@0.1.0"

# law_id в базе -> (короткое имя для ссылок, какие статьи брать; None = все)
LAWS = {
    "fz-197-2001": ("ТК РФ", None),
    # ГК РФ ч. 2: общие положения о подряде (гл. 37 §1) и возмездное оказание услуг (гл. 39)
    "fz-14-1996": ("ГК РФ", [str(a) for a in range(702, 730)] + [str(a) for a in range(779, 784)]),
    "fz-422-2018": ("422-ФЗ (самозанятые)", None),
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["npm", "pack", PACKAGE, "--pack-destination", tmp],
                       check=True, capture_output=True)
        tgz = next(Path(tmp).glob("*.tgz"))
        with tarfile.open(tgz) as tar:
            tar.extract("package/data/database.db", tmp, filter="data")
        db = sqlite3.connect(Path(tmp) / "package/data/database.db")

        rows = []
        for law_id, (short, articles) in LAWS.items():
            title, url = db.execute("select title, source_url from laws where id=?", (law_id,)).fetchone()
            for article, art_title, content in db.execute(
                    "select article, title, content from provisions where law_id=?", (law_id,)):
                if articles is not None and article not in articles:
                    continue
                rows.append({"law": short, "law_title": title, "source_url": url,
                             "article": article, "title": art_title, "text": content})
        db.close()

    def key(r):
        return (list(LAWS).index(next(k for k, v in LAWS.items() if v[0] == r["law"])),
                [int(x) for x in r["article"].split(".")])

    rows.sort(key=key)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Статей: {len(rows)} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
