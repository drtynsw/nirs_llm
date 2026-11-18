"""Обезличенная аналитика обращений для администратора (spec/tech/admin.md).

Текст вопроса и точный возраст не сохраняются: только тема, возрастная группа,
номера найденных статей и время ответа.
"""
import csv
import io
import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.age import GROUPS

NO_AGE = "не указан"
OTHER_TOPIC = "Другое"
TOPICS = [
    ("Самозанятость", r"самозан|налог|нпд"),
    ("Договор подряда и услуг", r"подряд|услуг|гпх|фриланс|заказчик"),
    ("Испытательный срок", r"испыт|стажир"),
    ("Увольнение", r"увол|расторж|сокращ"),
    ("Отпуска и учёба", r"отпуск|сесси|учебн|каникул"),
    ("Оплата труда", r"зарплат|заработн|плат|деньг|задерж|преми"),
    ("Рабочее время", r"час|смен|график|ноч|выходн|переработ|сверхуроч"),
    ("Оформление", r"оформ|договор|устро|официал|трудоустр"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    day TEXT NOT NULL,
    age_group TEXT,
    topic TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    n_docs INTEGER NOT NULL,
    top_docs TEXT NOT NULL,
    took_ms REAL NOT NULL,
    no_answer INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS events_day ON events(day);
"""
CSV_FIELDS = ("ts", "age_group", "topic", "top_k", "n_docs", "top_docs", "took_ms", "no_answer")


def topic(question: str) -> str:
    q = question.lower().replace("ё", "е")
    for name, pattern in TOPICS:
        if re.search(pattern, q):
            return name
    return OTHER_TOPIC


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def record(self, *, question: str, age_group: str | None, top_k: int,
               documents: list[dict], took_ms: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        top = [[d["doc_id"], d["citation"]] for d in documents[:3]]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO events (ts, day, age_group, topic, top_k, n_docs, top_docs, took_ms, no_answer)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now.isoformat(timespec="seconds"), now.date().isoformat(), age_group,
                 topic(question), top_k, len(documents),
                 json.dumps(top, ensure_ascii=False), took_ms, int(not documents)))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _rows(self, since: str) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._db.execute(
                "SELECT ts, age_group, topic, top_k, n_docs, top_docs, took_ms, no_answer"
                " FROM events WHERE day >= ? ORDER BY ts", (since,))
            return cur.fetchall()

    @staticmethod
    def _since(days: int, now: datetime | None) -> tuple[datetime, str]:
        today = (now or datetime.now(timezone.utc)).date()
        start = today - timedelta(days=days - 1)
        return today, start.isoformat()

    def stats(self, days: int = 30, now: datetime | None = None) -> dict:
        today, since = self._since(days, now)
        rows = self._rows(since)
        total = len(rows)
        no_answer = sum(r[7] for r in rows)

        per_day = {}
        ages, topics, cross, articles, citations = {}, {}, {}, {}, {}
        for ts, age_group, top, _k, _n, top_docs, _ms, _na in rows:
            per_day[ts[:10]] = per_day.get(ts[:10], 0) + 1
            g = age_group or NO_AGE
            ages[g] = ages.get(g, 0) + 1
            topics[top] = topics.get(top, 0) + 1
            cross[(g, top)] = cross.get((g, top), 0) + 1
            for doc_id, citation in json.loads(top_docs):
                articles[doc_id] = articles.get(doc_id, 0) + 1
                citations[doc_id] = citation

        by_day = []
        d = today - timedelta(days=days - 1)
        while d <= today:
            by_day.append({"date": d.isoformat(), "count": per_day.get(d.isoformat(), 0)})
            d += timedelta(days=1)

        order = list(GROUPS) + [NO_AGE]
        given = total - ages.get(NO_AGE, 0)
        return {
            "period_days": days,
            "since": since,
            "total": total,
            "no_answer": no_answer,
            "no_answer_share": round(no_answer / total, 3) if total else 0.0,
            "avg_took_ms": round(sum(r[6] for r in rows) / total, 1) if total else 0.0,
            "age_given_share": round(given / total, 3) if total else 0.0,
            "by_day": by_day,
            "by_age_group": [{"group": g, "count": ages.get(g, 0)} for g in order],
            "by_topic": [{"topic": t, "count": n}
                         for t, n in sorted(topics.items(), key=lambda x: (-x[1], x[0]))],
            "age_topic": [{"group": g, "topic": t, "count": n}
                          for (g, t), n in sorted(cross.items(),
                                                  key=lambda x: (order.index(x[0][0]), -x[1]))],
            "top_articles": [{"doc_id": i, "citation": citations[i], "count": n}
                             for i, n in sorted(articles.items(), key=lambda x: (-x[1], x[0]))[:10]],
        }

    def export_csv(self, days: int = 30, now: datetime | None = None) -> str:
        _, since = self._since(days, now)
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
        writer.writerow(CSV_FIELDS)
        for r in self._rows(since):
            top_docs = ",".join(doc_id for doc_id, _ in json.loads(r[5]))
            writer.writerow([r[0], r[1] or "", r[2], r[3], r[4], top_docs, r[6], r[7]])
        return "﻿" + buf.getvalue()
