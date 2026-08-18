"""Objective-aware fill-in-the-middle completion evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from ..data.fim import FIMSample
from ..model.generation import generate
from .compile import compile_typescript
from .fim import exact_match, token_accuracy
from .runner import repetition_rate
from .syntax import parse_typescript

FIMGenerator = Callable[[str, int], str]
TokenEncoder = Callable[[str], list[int]]
_CONTROL_MARKERS = ("<pad>", "<eos>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>")


def load_fim_tasks(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "scope", "tasks"}:
        raise ValueError("FIM task file has an unsupported schema")
    if raw["schema_version"] != 1 or raw["scope"] != "project-held-out-fim-fixtures":
        raise ValueError("FIM task file version or scope is unsupported")
    tasks = raw["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("FIM task file must contain tasks")
    required = {"id", "filename", "prefix", "suffix", "middle", "max_new_tokens"}
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != required:
            raise ValueError("FIM task has missing or unknown fields")
        task_id = task["id"]
        filename = task["filename"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("FIM task ids must be unique non-empty strings")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("FIM task filename must be a basename")
        if Path(filename).suffix not in {".ts", ".tsx"}:
            raise ValueError("FIM task filename must be TypeScript or TSX")
        for field in ("prefix", "suffix"):
            value = task[field]
            if not isinstance(value, str):
                raise ValueError(f"FIM task {field} must be a string")
            if any(marker in value for marker in _CONTROL_MARKERS):
                raise ValueError("FIM task source fields cannot contain control markers")
        middle = task["middle"]
        if not isinstance(middle, str) or not middle:
            raise ValueError("FIM task middle must be a non-empty string")
        if any(marker in middle for marker in _CONTROL_MARKERS):
            raise ValueError("FIM task source fields cannot contain control markers")
        maximum = task["max_new_tokens"]
        if not isinstance(maximum, int) or not 1 <= maximum <= 256:
            raise ValueError("FIM max_new_tokens must be between 1 and 256")
        seen.add(task_id)
    return tasks


def fim_prompt(task: dict[str, Any]) -> str:
    return f"<fim_prefix>{task['prefix']}<fim_suffix>{task['suffix']}<fim_middle>"


def evaluate_fim_tasks(
    tasks: list[dict[str, Any]],
    generator: FIMGenerator,
    encode: TokenEncoder,
    *,
    tool_root: str | Path | None = None,
    compile_timeout_seconds: int = 10,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        prompt = fim_prompt(task)
        completion = generator(prompt, task["max_new_tokens"])
        repeated = generator(prompt, task["max_new_tokens"])
        expected = task["middle"]
        reconstructed = task["prefix"] + completion + task["suffix"]
        expected_tokens = encode(expected)
        actual_tokens = encode(completion)
        if tool_root is None and compile_timeout_seconds == 10:
            parsed = parse_typescript(reconstructed, task["filename"])
            compiled = compile_typescript(reconstructed, task["filename"])
        else:
            parsed = parse_typescript(
                reconstructed,
                filename=task["filename"],
                timeout=compile_timeout_seconds,
                tool_root=tool_root,
            )
            compiled = compile_typescript(
                reconstructed,
                filename=task["filename"],
                timeout=compile_timeout_seconds,
                tool_root=tool_root,
            )
        results.append(
            {
                "id": task["id"],
                "filename": task["filename"],
                "prompt": prompt,
                "completion": completion,
                "expected_middle": expected,
                "reconstructed": reconstructed,
                "generation_characters": len(completion),
                "fim_exact_match": exact_match(expected_tokens, actual_tokens),
                "fim_token_accuracy": token_accuracy(expected_tokens, actual_tokens),
                "repetition_rate": repetition_rate(completion),
                "deterministic": completion == repeated,
                "syntax": parsed,
                "compile": compiled,
            }
        )
    count = len(results)
    return {
        "schema_version": 1,
        "scope": "project-held-out-fim-fixtures",
        "disclaimer": "Local project fixtures only; no claim of benchmark independence.",
        "task_count": count,
        "fim_exact_match_rate": sum(bool(item["fim_exact_match"]) for item in results) / count,
        "fim_token_accuracy": sum(float(item["fim_token_accuracy"]) for item in results) / count,
        "syntax_parse_rate": sum(bool(item["syntax"].get("success")) for item in results) / count,
        "compilation_rate": sum(bool(item["compile"].get("success")) for item in results) / count,
        "deterministic_rate": sum(bool(item["deterministic"]) for item in results) / count,
        "mean_repetition_rate": sum(float(item["repetition_rate"]) for item in results) / count,
        "tasks": results,
    }


def model_fim_generator(
    model: Any,
    tokenizer: Any,
    device: torch.device | str = "cpu",
    *,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    use_kv_cache: bool = True,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
) -> FIMGenerator:
    special_ids = {
        token_id
        for token in ("<pad>", "<eos>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>")
        if (token_id := tokenizer.token_to_id(token)) is not None
    }

    def complete(prompt: str, max_new_tokens: int) -> str:
        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            raise ValueError("tokenizer produced an empty FIM prompt")
        available = model.config.context_length - len(prompt_ids)
        if available <= 0:
            raise ValueError("FIM prompt exceeds model context length")
        count = min(max_new_tokens, available)
        output = generate(
            model,
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=count,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_ids=special_ids,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_kv_cache=use_kv_cache,
        )
        generated_ids = output[0, len(prompt_ids) :].tolist()
        generated_ids = [token for token in generated_ids if token not in special_ids]
        return tokenizer.decode(generated_ids, skip_special_tokens=False)

    return complete


def make_fim_sample(task: dict[str, Any]) -> FIMSample:
    start = len(task["prefix"])
    end = start + len(task["middle"])
    return FIMSample(
        serialized=fim_prompt(task) + task["middle"],
        prefix=task["prefix"],
        suffix=task["suffix"],
        middle=task["middle"],
        start=start,
        end=end,
    )
