import json
from pathlib import Path

import pytest
import torch

from ts_coder.evaluation import completion


def test_completion_fixture_is_versioned_and_held_out():
    tasks = completion.load_completion_tasks(Path("fixtures/evaluation/completion-tasks.json"))
    assert {task["id"] for task in tasks} == {
        "clamp-number",
        "interface-format",
        "class-counter",
        "async-result",
        "tsx-badge",
    }
    assert all(0 < task["max_new_tokens"] <= 256 for task in tasks)


def test_completion_evaluation_reports_per_task_and_aggregate(monkeypatch):
    monkeypatch.setattr(
        completion,
        "parse_typescript",
        lambda source, filename: {"success": filename.endswith(".ts"), "diagnostics": []},
    )
    monkeypatch.setattr(
        completion,
        "compile_typescript",
        lambda source, filename: {"success": "bad" not in source, "diagnostics": []},
    )
    calls = 0

    def generate(prompt: str, maximum: int) -> str:
        nonlocal calls
        calls += 1
        return "return 1;\n}\n" if "good" in prompt else "bad bad bad bad bad"

    tasks = [
        {"id": "good", "filename": "good.ts", "prompt": "good", "max_new_tokens": 8},
        {"id": "bad", "filename": "bad.tsx", "prompt": "bad", "max_new_tokens": 8},
    ]
    result = completion.evaluate_completion_tasks(tasks, generate)
    assert calls == 4
    assert result["scope"] == "project-held-out-fixtures"
    assert result["syntax_parse_rate"] == 0.5
    assert result["compilation_rate"] == 0.5
    assert result["deterministic_rate"] == 1.0
    assert result["tasks"][0]["syntax"]["success"] is True
    assert result["tasks"][1]["repetition_rate"] > 0


def test_completion_task_schema_rejects_unknown_fields(tmp_path):
    fixture = tmp_path / "tasks.json"
    fixture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "project-held-out-fixtures",
                "tasks": [
                    {
                        "id": "one",
                        "filename": "one.ts",
                        "prompt": "const one = ",
                        "max_new_tokens": 4,
                        "surprise": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing or unknown"):
        completion.load_completion_tasks(fixture)


def test_model_completion_generator_forwards_decode_controls(monkeypatch):
    calls = []

    class Tokenizer:
        def encode(self, value):
            return type("Encoded", (), {"ids": [1]})()

        def decode(self, values, skip_special_tokens=False):
            return "ok"

    class Model:
        config = type("Config", (), {"context_length": 32})()

    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[1, 2]])

    monkeypatch.setattr(completion, "generate", fake_generate)
    generator = completion.model_completion_generator(
        Model(), Tokenizer(), repetition_penalty=1.1, no_repeat_ngram_size=3
    )
    assert generator("prompt", 4) == "ok"
    assert calls[0]["repetition_penalty"] == 1.1
    assert calls[0]["no_repeat_ngram_size"] == 3
