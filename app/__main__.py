"""Запуск: python -m app [--port 8000]"""
import argparse

from app.server import run

parser = argparse.ArgumentParser(description="Консультант по трудовому праву")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
run(args.host, args.port)
