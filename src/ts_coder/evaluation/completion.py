"""Deterministic project-fixture completion evaluation.

These tasks are local regression fixtures, not an independent benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from ..model.generation import generate
from .compile import compile_typescript
from .runner import repetition_rate
from .syntax import parse_typescript

CompletionGenerator = Callable[[str, int], str]


def load_completion_tasks(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "scope", "tasks"}:
        raise ValueError("completion task file has an unsupported schema")
    if raw["schema_version"] != 1 or raw["scope"] != "project-held-out-fixtures":
        raise ValueError("completion task file version or scope is unsupported")
    tasks = raw["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("completion task file must contain tasks")
    required = {"id", "filename", "prompt", "max_new_tokens"}
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != required:
            raise ValueError("completion task has missing or unknown fields")
        task_id = task["id"]
        filename = task["filename"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("completion task ids must be unique non-empty strings")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("completion task filename must be a basename")
        if Path(filename).suffix not in {".ts", ".tsx"}:
            raise ValueError("completion task filename must be TypeScript or TSX")
        if not isinstance(task["prompt"], str) or not task["prompt"]:
            raise ValueError("completion task prompt must be non-empty")
        if not isinstance(task["max_new_tokens"], int) or not 1 <= task["max_new_tokens"] <= 256:
            raise ValueError("completion max_new_tokens must be between 1 and 256")
        seen.add(task_id)
    return tasks


def evaluate_completion_tasks(
    tasks: list[dict[str, Any]], generator: CompletionGenerator
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        completion = generator(task["prompt"], task["max_new_tokens"])
        repeated = generator(task["prompt"], task["max_new_tokens"])
        source = task["prompt"] + completion
        parsed = parse_typescript(source, filename=task["filename"])
        compiled = compile_typescript(source, filename=task["filename"])
        results.append(
            {
                "id": task["id"],
                "filename": task["filename"],
                "completion": completion,
                "generation_characters": len(completion),
                "repetition_rate": repetition_rate(completion),
                "deterministic": completion == repeated,
                "syntax": parsed,
                "compile": compiled,
            }
        )
    count = len(results)
    return {
        "schema_version": 1,
        "scope": "project-held-out-fixtures",
        "disclaimer": "Local project fixtures only; no claim of benchmark independence.",
        "task_count": count,
        "syntax_parse_rate": sum(bool(item["syntax"].get("success")) for item in results) / count,
        "compilation_rate": sum(bool(item["compile"].get("success")) for item in results) / count,
        "deterministic_rate": sum(bool(item["deterministic"]) for item in results) / count,
        "mean_repetition_rate": sum(float(item["repetition_rate"]) for item in results) / count,
        "tasks": results,
    }


def model_completion_generator(
    model: Any,
    tokenizer: Any,
    device: torch.device | str = "cpu",
    *,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> CompletionGenerator:
    """Create a greedy generator with explicit, reportable decode controls."""

    def complete(prompt: str, max_new_tokens: int) -> str:
        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            raise ValueError("tokenizer produced an empty prompt")
        available = model.config.context_length - len(prompt_ids)
        if available <= 0:
            raise ValueError("completion prompt exceeds model context length")
        count = min(max_new_tokens, available)
        output = generate(
            model,
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=count,
            temperature=0.0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return tokenizer.decode(output[0, len(prompt_ids) :].tolist(), skip_special_tokens=False)

    return complete
