import json
from pathlib import Path

import pytest
import torch

from ts_coder.evaluation import fim_completion


def test_fim_fixture_is_versioned_and_reconstructable():
    tasks = fim_completion.load_fim_tasks(Path("fixtures/evaluation/fim-tasks.json"))
    assert len(tasks) == 5
    for task in tasks:
        prompt = fim_completion.fim_prompt(task)
        assert prompt.startswith("<fim_prefix>")
        assert fim_completion.make_fim_sample(task).prefix + task["middle"] + task["suffix"]


def test_fim_evaluation_reports_reconstruction_metrics(monkeypatch):
    monkeypatch.setattr(
        fim_completion,
        "parse_typescript",
        lambda source, filename: {"success": "good" in source, "diagnostics": []},
    )
    monkeypatch.setattr(
        fim_completion,
        "compile_typescript",
        lambda source, filename: {"success": "bad" not in source, "diagnostics": []},
    )
    tasks = [
        {
            "id": "good",
            "filename": "good.ts",
            "prefix": "good(",
            "suffix": ")",
            "middle": "value",
            "max_new_tokens": 8,
        },
        {
            "id": "bad",
            "filename": "bad.ts",
            "prefix": "bad(",
            "suffix": ")",
            "middle": "value",
            "max_new_tokens": 8,
        },
    ]

    def generate(prompt: str, maximum: int) -> str:
        return "value" if "good(" in prompt else "bad bad bad"

    result = fim_completion.evaluate_fim_tasks(tasks, generate, lambda value: value.split())
    assert result["fim_exact_match_rate"] == 0.5
    assert result["fim_token_accuracy"] == 0.5
    assert result["syntax_parse_rate"] == 0.5
    assert result["compilation_rate"] == 0.5
    assert result["deterministic_rate"] == 1.0
    assert result["tasks"][0]["reconstructed"] == "good(value)"


def test_fim_task_schema_rejects_unknown_fields(tmp_path):
    fixture = tmp_path / "tasks.json"
    fixture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "project-held-out-fim-fixtures",
                "tasks": [
                    {
                        "id": "one",
                        "filename": "one.ts",
                        "prefix": "const one = ",
                        "suffix": ";",
                        "middle": "1",
                        "max_new_tokens": 4,
                        "surprise": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing or unknown"):
        fim_completion.load_fim_tasks(fixture)


def test_fim_task_rejects_control_markers(tmp_path):
    fixture = tmp_path / "tasks.json"
    fixture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "project-held-out-fim-fixtures",
                "tasks": [
                    {
                        "id": "one",
                        "filename": "one.ts",
                        "prefix": "const one = ",
                        "suffix": ";",
                        "middle": "<fim_middle>",
                        "max_new_tokens": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="control markers"):
        fim_completion.load_fim_tasks(fixture)


def test_fim_task_allows_file_boundary_spans(tmp_path):
    fixture = tmp_path / "tasks.json"
    fixture.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "project-held-out-fim-fixtures",
                "tasks": [
                    {
                        "id": "one",
                        "filename": "one.ts",
                        "prefix": "",
                        "suffix": "",
                        "middle": "const one = 1;",
                        "max_new_tokens": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert fim_completion.load_fim_tasks(fixture)[0]["prefix"] == ""


def test_model_fim_generator_excludes_control_marker(monkeypatch):
    class Tokenizer:
        ids = {"<pad>": 1, "<eos>": 2, "<fim_prefix>": 3, "<fim_suffix>": 4, "<fim_middle>": 5}

        def token_to_id(self, token):
            return self.ids.get(token)

        def encode(self, value):
            return type("Encoded", (), {"ids": [9] if value else []})()

        def decode(self, values, skip_special_tokens=False):
            return "body" if values == [7] else ""

    class Model:
        config = type("Config", (), {"context_length": 32})()

    def fake_generate(model, input_ids, **kwargs):
        assert kwargs["stop_ids"] == {1, 2, 3, 4, 5}
        return torch.tensor([[9, 7, 2]])

    monkeypatch.setattr(fim_completion, "generate", fake_generate)
    generator = fim_completion.model_fim_generator(Model(), Tokenizer())
    assert generator("prompt", 4) == "body"
