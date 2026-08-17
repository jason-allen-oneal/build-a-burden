import pytest

from ts_coder.training.authorization import (
    TrainingAuthorizationError,
    append_training_authorization,
    make_training_authorization,
    require_training_authorization,
)


def write(path, value):
    path.write_text(value, encoding="utf-8")
    return path


def test_exact_training_authorization_binds_every_artifact(tmp_path) -> None:
    training = write(tmp_path / "training.yaml", "run: approved\n")
    model = write(tmp_path / "model.yaml", "model: tiny\n")
    manifest = write(tmp_path / "manifest.jsonl", "{}\n")
    corpus = write(tmp_path / "documents.jsonl", '{"record":{},"text":"x"}\n')
    tokenizer = write(tmp_path / "tokenizer.json", "tokenizer\n")
    ledger = tmp_path / "authorizations.jsonl"
    record = make_training_authorization(
        run_name="approved-run",
        training_config=training,
        model_config=model,
        manifest=manifest,
        corpus=corpus,
        tokenizer=tokenizer,
        max_tokens=2_000_000,
        approved_by="reviewer",
        approved_at="2026-08-17T12:00:00Z",
    )
    append_training_authorization(ledger, record)

    approved = require_training_authorization(
        ledger,
        run_name="approved-run",
        training_config=training,
        model_config=model,
        manifest=manifest,
        corpus=corpus,
        tokenizer=tokenizer,
        max_tokens=2_000_000,
    )

    assert approved.authorization_id == record.authorization_id

    model.write_text("model: changed\n", encoding="utf-8")
    with pytest.raises(TrainingAuthorizationError, match="no exact authorization"):
        require_training_authorization(
            ledger,
            run_name="approved-run",
            training_config=training,
            model_config=model,
            manifest=manifest,
            corpus=corpus,
            tokenizer=tokenizer,
            max_tokens=2_000_000,
        )
