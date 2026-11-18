"""Benchmark поиска и отказов (spec/tech/rag.md, раздел «Проверка качества»).

Запуск:
    python -m benchmark.run                     # текущая конфигурация приложения
    python -m benchmark.run --mode stem --k1 1.5 --b 0.5 --no-expand
    python -m benchmark.run --grid              # сравнение конфигураций
    python -m benchmark.run --save              # сохранить отчёт в benchmark/results/

Тот же расчёт вызывается из страницы администратора (POST /api/admin/benchmark),
чтобы прогнать benchmark во время сдачи прямо в работающем приложении.

Метрики (методичка НИРС и занятие 3):
    Recall@k — доля релевантных статей в первых k, среднее по обычным вопросам;
    MRR@10   — средний обратный ранг первой релевантной статьи (0, если её нет в top-10);
    Correct Rejection Rate — доля ловушек, на которые система отказалась отвечать;
    False Rejection Rate — доля обычных вопросов, на которые система ошибочно отказала;
    Accuracy — доля полностью правильных ответов; считается вручную после подключения LLM.
Ловушки в метрики поиска не входят.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app import llm, rag, retrieval
from app.retrieval.corpus import by_id

ROOT = Path(__file__).resolve().parent
QUERIES = ROOT / "queries.jsonl"
RESULTS = ROOT / "results"
KS = (5, 10)
MRR_DEPTH = 10
TOP_K_ANSWER = 5   # столько статей получает генератор ответа, как в интерфейсе по умолчанию
LLM_WORKERS = 4    # параллельные вызовы модели при --llm


def load_queries(path: Path = QUERIES) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def matches(doc_id: str, relevant: str) -> bool:
    """Релевантная метка — doc_id или статья целиком: «tk_81» покрывает фрагменты tk_81_1, tk_81_2."""
    return doc_id == relevant or doc_id.startswith(relevant + "_")


def recall_at(ranked: list[str], relevant: list[str], k: int) -> float:
    top = ranked[:k]
    found = sum(1 for r in relevant if any(matches(d, r) for d in top))
    return found / len(relevant)


def first_rank(ranked: list[str], relevant: list[str]) -> int | None:
    for i, d in enumerate(ranked, 1):
        if any(matches(d, r) for r in relevant):
            return i
    return None


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _metrics(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in KS:
        out[f"recall@{k}"] = _mean([r[f"recall@{k}"] for r in rows])
    out[f"mrr@{MRR_DEPTH}"] = _mean([1 / r["rank"] if r["rank"] and r["rank"] <= MRR_DEPTH else 0.0 for r in rows])
    return out


def relevant_articles(relevant: list[str]) -> set[tuple[str, str]]:
    """Номера статей для меток релевантности (для сверки со ссылками в ответе)."""
    docs = by_id()
    return {ref for i, d in docs.items() if any(matches(i, r) for r in relevant)
            for ref in llm.citations(d["citation"])}


def answer_correct(row: dict, relevant: list[str]) -> bool:
    """Accuracy (авто): ответ есть, ссылается на релевантную статью и не ссылается на статьи вне выдачи."""
    if row["mode"] != "rag" or row["refused"] or row["unsupported_citations"]:
        return False
    supported, missing = llm.check_citations(row["answer"], row["top"][:llm.MAX_DOCS])
    return not missing and bool(supported & relevant_articles(relevant))


def evaluate(config: retrieval.Config = retrieval.DEFAULT, queries: list[dict] | None = None,
             settings: llm.Settings | None = None) -> dict:
    """settings=None — без LLM (быстро, ответы-заглушки); иначе ответы модели."""
    queries = queries if queries is not None else load_queries()
    settings = settings if settings is not None else rag.DISABLED
    t0 = time.perf_counter()
    r = retrieval.get(config)
    depth = max(max(KS), MRR_DEPTH)
    with ThreadPoolExecutor(max_workers=LLM_WORKERS if settings.enabled else 1) as pool:
        responses = list(pool.map(lambda q: rag.respond(q["query"], TOP_K_ANSWER, config, settings=settings),
                                  queries))

    regular, traps = [], []
    for q, resp in zip(queries, responses):
        ranked_docs = r.search(q["query"], depth)
        ranked = [d["doc_id"] for d in ranked_docs]
        row = {
            "id": q["id"], "query": q["query"], "type": q["type"], "style": q["style"],
            "split": q.get("split"), "topic": q["topic"],
            "top": [{"doc_id": d["doc_id"], "citation": d["citation"], "score": d["score"],
                     "relevant": any(matches(d["doc_id"], x) for x in q["relevant"])}
                    for d in ranked_docs[:5]],
            "top_score": ranked_docs[0]["score"] if ranked_docs else 0.0,
            "refused": resp["refused"],
            "answer": resp["answer"],
            "mode": resp["mode"],
            "llm_error": resp["llm_error"],
            "unsupported_citations": resp["unsupported_citations"],
            "reference": q["reference"],
        }
        if q["type"] == "trap":
            traps.append(row)
            continue
        row["relevant"] = q["relevant"]
        row["rank"] = first_rank(ranked, q["relevant"])
        for k in KS:
            row[f"recall@{k}"] = round(recall_at(ranked, q["relevant"], k), 4)
        row["missing@5"] = [x for x in q["relevant"] if not any(matches(d, x) for d in ranked[:5])]
        if settings.enabled:
            row["correct"] = answer_correct(row, q["relevant"])
        regular.append(row)

    by = lambda key: {v: _metrics([x for x in regular if x[key] == v])  # noqa: E731
                      for v in sorted({x[key] for x in regular if x[key]})}
    retrieval_metrics = {"all": _metrics(regular), "style": by("style"), "split": by("split"), "topic": by("topic")}
    crr = _mean([1.0 if t["refused"] else 0.0 for t in traps])
    frr = _mean([1.0 if x["refused"] else 0.0 for x in regular])
    answered = [x for x in regular + traps if not x["refused"]]
    llm_errors = sum(1 for x in regular + traps if x["llm_error"])
    if settings.enabled:
        accuracy = _mean([1.0 if x["correct"] else 0.0 for x in regular])
        note = "авто: ответ ссылается на релевантную статью и только на статьи выдачи; сверить вручную по reference"
    else:
        accuracy, note = None, "нужны ответы LLM: запустите с --llm (или галочкой на /admin)"
    errors = sorted((x for x in regular if not x["rank"] or x["rank"] > 3 or x["refused"]),
                    key=lambda x: (x["rank"] or 99))
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": config.as_dict(),
        "queries": {"regular": len(regular), "traps": len(traps)},
        "metrics": {
            **{key: retrieval_metrics["all"][key] for key in retrieval_metrics["all"] if key != "n"},
            "correct_rejection_rate": crr,
            "false_rejection_rate": frr,
            "accuracy": accuracy,
            "accuracy_note": note,
            "unsupported_citation_rate": _mean([1.0 if x["unsupported_citations"] else 0.0 for x in answered]),
            "llm_errors": llm_errors,
        },
        "llm": settings.describe() if settings.enabled else "не использовалась",
        "retrieval": retrieval_metrics,
        "errors": [{"id": x["id"], "query": x["query"], "rank": x["rank"], "refused": x["refused"],
                    "missing@5": x["missing@5"], "top": [t["citation"] for t in x["top"][:3]]} for x in errors],
        "regular": regular,
        "traps": traps,
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def save(report: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    c = report["config"]
    name = (f"{report['created'][:10]}_{c['mode_effective']}_k1-{c['k1']}_b-{c['b']}"
            f"_{'exp' if c['expand'] else 'noexp'}{'_llm' if report['metrics']['accuracy'] is not None else ''}.json")
    path = RESULTS / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def summary_line(report: dict) -> str:
    m, c = report["metrics"], report["config"]
    return (f"{c['mode_effective']:<5} k1={c['k1']:<4} b={c['b']:<4} exp={'да ' if c['expand'] else 'нет'} "
            f"| R@5 {m['recall@5']:.3f}  R@10 {m['recall@10']:.3f}  MRR {m['mrr@10']:.3f} "
            f"| CRR {m['correct_rejection_rate']:.2f}  FRR {m['false_rejection_rate']:.2f}")


def print_report(report: dict) -> None:
    m = report["metrics"]
    print(f"Конфигурация: {report['config']}")
    print(f"Запросов: {report['queries']['regular']} обычных, {report['queries']['traps']} ловушек; "
          f"{report['took_ms']} мс\n")
    print(f"Recall@5 {m['recall@5']:.3f}   Recall@10 {m['recall@10']:.3f}   MRR@10 {m['mrr@10']:.3f}")
    print(f"Correct Rejection Rate {m['correct_rejection_rate']:.2f}   "
          f"False Rejection Rate {m['false_rejection_rate']:.2f}")
    acc = f"{m['accuracy']:.3f}" if m["accuracy"] is not None else "—"
    print(f"Accuracy {acc} ({m['accuracy_note']})")
    print(f"LLM: {report['llm']}; ответов со ссылками вне выдачи {m['unsupported_citation_rate']:.2f}; "
          f"ошибок модели {m['llm_errors']}\n")
    if m["accuracy"] is not None:
        print("Ответы модели (+ — засчитан в авто-Accuracy):")
        for x in report["regular"]:
            print(f"  {'+' if x.get('correct') else '-'} {x['id']} {x['answer'][:160]}")
    for key in ("style", "split", "topic"):
        for v, mm in report["retrieval"][key].items():
            print(f"  {key:<6} {v:<20} n={mm['n']:<3} R@5 {mm['recall@5']:.3f}  R@10 {mm['recall@10']:.3f}  "
                  f"MRR {mm['mrr@10']:.3f}")
    print("\nЛовушки:")
    for t in report["traps"]:
        print(f"  {t['id']} {'отказ' if t['refused'] else 'ОТВЕТ'}  score={t['top_score']:.1f}  "
              f"top={t['top'][0]['citation'] if t['top'] else '—'}  {t['query']}")
    print("\nРазбор ошибок (первая релевантная ниже 3-го места, не найдена или отказ):")
    for e in report["errors"]:
        print(f"  {e['id']} rank={e['rank']} refused={e['refused']} нет в top-5: {', '.join(e['missing@5'])}"
              f" | выдача: {'; '.join(e['top'])} | {e['query']}")


def grid() -> list[dict]:
    """Меняем одну настройку за раз (занятие 3): нормализация -> расширение -> k1, b."""
    reports = []
    d = retrieval.DEFAULT
    print(f"Базовая конфигурация: {d.as_dict()}")
    print(f"CRR/FRR считаются при пороге {d.min_score}, подобранном для итоговой конфигурации;\n"
          "в других конфигурациях оценки в другом масштабе — сравнивайте по Recall и MRR.\n")
    print("1. Нормализация (без расширения запроса)")
    for mode in ("none", "stem", "lemma"):
        rep = evaluate(retrieval.Config(mode=mode, expand=False))
        reports.append(rep)
        print("  ", summary_line(rep))
    print("2. Расширение запроса (лемматизация) — смотрим на контрольной части")
    for exp in (False, True):
        rep = evaluate(retrieval.Config(expand=exp))
        reports.append(rep)
        ctrl, train = rep["retrieval"]["split"]["control"], rep["retrieval"]["split"]["train"]
        print("  ", summary_line(rep),
              f"| control R@5 {ctrl['recall@5']:.3f} MRR {ctrl['mrr@10']:.3f}"
              f" | train R@5 {train['recall@5']:.3f} MRR {train['mrr@10']:.3f}")
    print(f"3. Параметры BM25 (лемматизация, с расширением, вес названия {d.title_weight})")
    for k1 in (1.2, 1.5, 2.0):
        for b in (0.5, 0.75, 1.0):
            rep = evaluate(retrieval.Config(k1=k1, b=b))
            reports.append(rep)
            print("  ", summary_line(rep))
    return reports


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark поиска BM25")
    p.add_argument("--mode", choices=("none", "stem", "lemma"), default=retrieval.DEFAULT.mode)
    p.add_argument("--k1", type=float, default=retrieval.DEFAULT.k1)
    p.add_argument("--b", type=float, default=retrieval.DEFAULT.b)
    p.add_argument("--no-expand", action="store_true", help="без словаря расширения запроса")
    p.add_argument("--title-weight", type=int, default=retrieval.DEFAULT.title_weight)
    p.add_argument("--min-score", type=float, default=retrieval.DEFAULT.min_score,
                   help="порог С4: оценка лучшего документа")
    p.add_argument("--llm", action="store_true",
                   help="получить ответы LLM (LLM_BASE_URL, LLM_MODEL) и посчитать Accuracy")
    p.add_argument("--grid", action="store_true", help="сравнить конфигурации")
    p.add_argument("--save", action="store_true", help="сохранить отчёт(ы) в benchmark/results/")
    args = p.parse_args(argv)

    if args.grid:
        reports = grid()
        if args.save:
            path = RESULTS / f"{datetime.now():%Y-%m-%d}_grid.json"
            RESULTS.mkdir(exist_ok=True)
            path.write_text(json.dumps([{"config": r["config"], "metrics": r["metrics"],
                                         "retrieval": r["retrieval"]} for r in reports],
                                       ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\nСохранено: {path}")
        return

    config = retrieval.Config(mode=args.mode, k1=args.k1, b=args.b, expand=not args.no_expand,
                              title_weight=args.title_weight, min_score=args.min_score)
    settings = None
    if args.llm:
        from app import config as app_config
        app_config.load_dotenv()
        settings = llm.Settings.from_env()
        if not settings.enabled:
            p.error("--llm: задайте LLM_BASE_URL и LLM_MODEL (переменные окружения или .env)")
    report = evaluate(config, settings=settings)
    print_report(report)
    if args.save:
        print(f"\nСохранено: {save(report)}")


if __name__ == "__main__":
    main()
