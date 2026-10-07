# Архитектура

## Путь запроса

```
Браузер (app/static/index.html)
   │  POST /api/ask {question, top_k}
   ▼
app/server.py ── ask()
   │ 1. проверка ввода
   │ 2. search(question, top_k)  ──►  app/retrieval/  (КТ2: BM25)
   │                                   сейчас: app/search_stub.py
   │ 3. answer(question, docs)   ──►  app/llm.py      (КТ2: вызов LLM)
   │                                   сейчас: app/llm_stub.py
   ▼
JSON {answer, documents[], took_ms, mode}
```
