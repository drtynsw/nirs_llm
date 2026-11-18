"""Настройки из переменных окружения и файла .env (без внешних библиотек).

| Переменная | По умолчанию |
| ADMIN_TOKEN | случайный при старте, печатается в консоль |
| ANALYTICS_DB | data/analytics.sqlite3 |
"""
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Читает KEY=VALUE из .env; уже заданные переменные окружения не перезаписывает."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def admin_token() -> tuple[str, bool]:
    """(токен, сгенерирован_ли_он_сейчас)."""
    token = os.environ.get("ADMIN_TOKEN", "").strip()
    if token:
        return token, False
    return secrets.token_urlsafe(18), True


def analytics_db() -> str:
    return os.environ.get("ANALYTICS_DB", "").strip() or str(ROOT / "data" / "analytics.sqlite3")
