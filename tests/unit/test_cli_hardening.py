import pytest

from ts_coder.cli import _generation_settings, _read_evaluation_config


def test_evaluation_config_preserves_declared_contract(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "split: test",
                "max_evaluation_tokens: 4096",
                "generation: {temperature: 0.2, top_k: 20, top_p: 0.9, use_kv_cache: false}",
                "metrics: [cross_entropy, compilation_rate]",
                "typescript_tool: tools/typescript",
                "compile_timeout_seconds: 7",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = _read_evaluation_config(path)
    generation = _generation_settings(config)

    assert config["split"] == "test"
    assert config["max_evaluation_tokens"] == 4096
    assert config["compile_timeout_seconds"] == 7
    assert generation == {
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.9,
        "use_kv_cache": False,
    }


def test_evaluation_config_rejects_ignored_fields(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text("schema_version: 1\nnot_used: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown evaluation"):
        _read_evaluation_config(path)
