"""Lightweight JSONL trajectory logger + rich console."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from rich.console import Console

_console = Console()


def log(msg: str, style: str = "") -> None:
    _console.print(msg, style=style)


def info(msg: str) -> None:
    log(f"[bold cyan]ℹ[/] {msg}")


def warn(msg: str) -> None:
    log(f"[bold yellow]⚠[/] {msg}")


def err(msg: str) -> None:
    log(f"[bold red]✗[/] {msg}")


def ok(msg: str) -> None:
    log(f"[bold green]✓[/] {msg}")


class JsonlWriter:
    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, mode, encoding="utf-8")

    def write(self, obj: Any) -> None:
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        self.f.write(json.dumps(obj, default=str) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
