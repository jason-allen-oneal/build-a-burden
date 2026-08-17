"""Command-line entry points for the local vertical slice."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from itertools import cycle, islice
from pathlib import Path

import torch
import yaml

from .corpus.acquire import acquire_git_source, verify_snapshot_receipt
from .corpus.approval import (
    append_approval,
    load_approval_manifest,
    make_approval,
    require_approved_source,
)
from .corpus.deduplicate import assign_clusters
from .corpus.ingest import ingest_repository, sanitize_source_uri
from .corpus.licensing import DEFAULT_ACCEPTED, detect_license
from .corpus.manifest import write_manifest
from .corpus.split import assign_splits
from .data.causal import causal_example, causal_examples
from .data.fim import make_fim
from .data.objectives import use_fim
from .data.packed_stream import PackedTokenBlockBatcher
from .data.streaming import DataCursor, ShardDescriptor, ShardManifest, StreamingShardDataset
from .data.token_stream import TokenizedStreamingBatcher, TokenizedStreamingDataset
from .evaluation.completion import (
    evaluate_completion_tasks,
    load_completion_tasks,
    model_completion_generator,
)
from .evaluation.fim import exact_match, token_accuracy
from .evaluation.fim_completion import (
    evaluate_fim_tasks,
    load_fim_tasks,
    model_fim_generator,
)
from .evaluation.loss import perplexity
from .evaluation.runner import evaluate_sources
from .model import count_parameters, model_audit
from .model.config import ModelConfig as RuntimeModelConfig
from .model.generation import generate
from .model.transformer import Transformer
from .reproducibility import create_run_dir, seed_everything, sha256_file
from .tokenizer.metrics import evaluate_tokenizer
from .tokenizer.trainer import load_tokenizer, train_tokenizer
from .training.checkpoint import load_checkpoint_payload
from .training.distributed import validate_distributed_config
from .training.metrics import input_token_count
from .training.optimizer import build_adamw
from .training.scheduler import build_cosine_scheduler
from .training.trainer import Trainer


def _cmd_generate(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    payload = load_checkpoint_payload(checkpoint)
    metadata = payload.get("metadata", {})
    raw_config = metadata.get("resolved_config", {})
    if "model" in raw_config:
        raw_config = raw_config["model"]
    required = {
        name: raw_config[name]
        for name in (
            "vocab_size",
            "context_length",
            "layers",
            "hidden_size",
            "attention_heads",
            "kv_heads",
            "ffn_size",
        )
    }
    model = Transformer(RuntimeModelConfig(**required))
    model.load_state_dict(payload["model"])
    model.eval()
    # A tokenizer artifact is optional for the development CLI; byte fallback is deterministic.
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    seed_everything(args.seed)
    tokenizer = None
    tokenizer_path = (
        Path(args.tokenizer)
        if args.tokenizer
        else checkpoint.parents[3] / "tokenizers" / "dev" / "tokenizer.json"
    )
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
    prompt_ids = tokenizer.encode(prompt).ids if tokenizer else [b for b in prompt.encode("utf-8")]
    ids = torch.tensor([prompt_ids], dtype=torch.long) % model.config.vocab_size
    out = generate(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    new_ids = out[0, ids.shape[1] :].tolist()
    generated = (
        tokenizer.decode(new_ids, skip_special_tokens=False)
        if tokenizer
        else bytes(int(x) % 256 for x in new_ids).decode("utf-8", errors="replace")
    )
    print(
        json.dumps(
            {"checkpoint": str(checkpoint), "seed": args.seed, "tokens": args.max_new_tokens},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    sys.stdout.write(generated)
    return 0


_DATA_KEYS = {
    "schema_version",
    "seed",
    "sources",
    "allowed_extensions",
    "accepted_licenses",
    "output_manifest",
    "output_corpus",
    "max_file_bytes",
    "max_files",
    "max_total_bytes",
    "max_path_depth",
    "splits",
    "near_duplicate_threshold",
    "retrieved_at",
    "append_manifest",
    "pipeline_version",
    "approval_manifest",
    "approval_manifest_sha256",
    "require_approval",
    "acquisition_root",
    "acquisition_timeout_seconds",
    "acquisition_max_file_bytes",
    "acquisition_max_files",
    "acquisition_max_total_bytes",
    "acquisition_max_path_depth",
}
_TOKENIZER_KEYS = {
    "schema_version",
    "type",
    "vocab_size",
    "min_frequency",
    "seed",
    "corpus_manifest",
    "corpus_artifact",
    "output_dir",
    "special_tokens",
}
_MODEL_KEYS = {
    "schema_version",
    "name",
    "vocab_size",
    "context_length",
    "layers",
    "hidden_size",
    "attention_heads",
    "kv_heads",
    "ffn_size",
    "normalization",
    "position_encoding",
    "activation",
    "tied_embeddings",
    "dropout",
    "rope_theta",
    "rms_norm_eps",
    "initializer_range",
    "use_sdpa",
    "gradient_checkpointing",
    "training_tokens",
    "planning_only",
    "minimum_parameters",
    "architecture",
    "requires_recalculation",
}
_TRAINING_KEYS = {
    "schema_version",
    "run_name",
    "model_config",
    "tokenizer_dir",
    "manifest",
    "corpus_artifact",
    "output_root",
    "seed",
    "device",
    "precision",
    "sequence_length",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "max_tokens",
    "optimizer",
    "scheduler",
    "gradient_clip_norm",
    "objectives",
    "validation_interval_tokens",
    "validation_max_tokens",
    "checkpoint_interval_tokens",
    "sample_interval_tokens",
    "streaming_tokens_per_step_estimate",
    "packed_streaming",
    "exclude_compiler_harness",
    "approval_required",
    "resume_from",
    "distributed",
    "streaming",
    "shard_manifest",
    "max_line_bytes",
}
_EVALUATION_KEYS = {
    "schema_version",
    "seed",
    "manifest",
    "split",
    "max_new_tokens",
    "generation",
    "metrics",
    "typescript_tool",
    "compile_timeout_seconds",
    "output",
}

_NESTED_KEYS = {
    "sources": {"path", "receipt", "source_id", "source_uri", "commit_sha"},
    "splits": {"train", "validation", "test"},
    "optimizer": {"name", "learning_rate", "betas", "weight_decay"},
    "scheduler": {"name", "warmup_tokens", "minimum_learning_rate_ratio"},
    "objectives": {"causal_fraction", "fim_fraction", "fim_min_span", "fim_max_span"},
    "distributed": {
        "strategy",
        "backend",
        "world_size",
        "rank",
        "local_rank",
        "timeout_seconds",
        "data_partition",
    },
    "generation": {"temperature", "top_k", "top_p", "use_kv_cache"},
    "architecture": {
        "vocab_size",
        "context_length",
        "layers",
        "hidden_size",
        "attention_heads",
        "kv_heads",
        "ffn_size",
        "normalization",
        "position_encoding",
        "activation",
        "tied_embeddings",
        "parameter_count_estimate",
        "hardware_contract",
        "gradient_checkpointing",
    },
}


def _cmd_evaluate(args: argparse.Namespace) -> int:
    evaluation_cfg = _read_yaml(args.config, _EVALUATION_KEYS) if args.config else {}
    if args.output is None and evaluation_cfg.get("output"):
        args.output = evaluation_cfg["output"]
    sources = [Path(args.source).read_text(encoding="utf-8")] if args.source else []
    corpus: list[str] = []
    generated_token_count = 0
    deterministic_generation: bool | None = None
    validation_cross_entropy: float | None = None
    fim_exact: float | None = None
    fim_accuracy: float | None = None
    generation_tokens_per_second: float | None = None
    generation_cfg = evaluation_cfg.get("generation", {})
    max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 32))
    temperature = float(generation_cfg.get("temperature", 0.0))
    top_k = int(generation_cfg.get("top_k", 0))
    top_p = float(generation_cfg.get("top_p", 1.0))
    if max_new_tokens <= 0 or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid evaluation generation settings")
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else Path("artifacts/runs/dev/checkpoints/latest")
    )
    if checkpoint.exists() and not sources:
        payload = load_checkpoint_payload(checkpoint)
        raw_config = payload.get("metadata", {}).get("resolved_config", {}).get("model", {})
        runtime_fields = set(RuntimeModelConfig.__dataclass_fields__)
        model = Transformer(
            RuntimeModelConfig(
                **{
                    k: v
                    for k, v in raw_config.items()
                    if k != "schema_version" and k in runtime_fields
                }
            )
        )
        model.load_state_dict(payload["model"])
        model.eval()
        documents = _corpus_documents(Path.cwd())
        corpus = [item["text"] for item in documents]
        prompt = documents[0]["text"][:96] if documents else "export const value = "
        tokenizer_path = checkpoint.parents[3] / "tokenizers" / "dev" / "tokenizer.json"
        tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else None
        prompt_ids = tokenizer.encode(prompt).ids if tokenizer else list(prompt.encode())
        generation_started = time.perf_counter()
        generated_ids = generate(
            model,
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k or None,
            top_p=top_p,
        )[0].tolist()
        generation_elapsed = max(time.perf_counter() - generation_started, 1e-9)
        repeated_ids = generate(
            model,
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k or None,
            top_p=top_p,
        )[0].tolist()
        deterministic_generation = generated_ids == repeated_ids
        generated_token_count = max(0, len(generated_ids) - len(prompt_ids))
        generation_tokens_per_second = generated_token_count / generation_elapsed
        sources = [
            tokenizer.decode(generated_ids, skip_special_tokens=False)
            if tokenizer
            else bytes(generated_ids).decode("utf-8", errors="replace")
        ]
        evaluation_docs = [
            item for item in documents if item["record"].get("split") == "validation"
        ] or documents
        validation_losses: list[float] = []
        fim_matches: list[bool] = []
        fim_accuracies: list[float] = []
        with torch.no_grad():
            for item in evaluation_docs:
                text = item["text"]
                ids = tokenizer.encode(text).ids if tokenizer else list(text.encode())
                if len(ids) >= 2:
                    example = causal_example(ids, model.config.context_length, 0)
                    output = model(
                        **{
                            key: torch.tensor([value], dtype=torch.long)
                            for key, value in example.items()
                        }
                    )
                    if output.loss is not None and torch.isfinite(output.loss):
                        validation_losses.append(float(output.loss))
                if tokenizer:
                    try:
                        sample = make_fim(
                            text,
                            item["record"]["record_id"],
                            int(evaluation_cfg.get("seed", 42)),
                            4,
                            64,
                        )
                    except ValueError:
                        continue
                    fim_prompt = (
                        f"<fim_prefix>{sample.prefix}<fim_suffix>{sample.suffix}<fim_middle>"
                    )
                    prompt_tokens = tokenizer.encode(fim_prompt).ids
                    expected_tokens = tokenizer.encode(sample.middle).ids
                    if (
                        expected_tokens
                        and len(prompt_tokens) + len(expected_tokens) <= model.config.context_length
                    ):
                        predicted = generate(
                            model,
                            torch.tensor([prompt_tokens], dtype=torch.long),
                            max_new_tokens=len(expected_tokens),
                        )[0, len(prompt_tokens) :].tolist()
                        fim_matches.append(exact_match(expected_tokens, predicted))
                        fim_accuracies.append(token_accuracy(expected_tokens, predicted))
        if validation_losses:
            validation_cross_entropy = sum(validation_losses) / len(validation_losses)
        if fim_matches:
            fim_exact = sum(fim_matches) / len(fim_matches)
            fim_accuracy = sum(fim_accuracies) / len(fim_accuracies)
    if not corpus:
        corpus_path = Path.cwd() / "artifacts/corpus/dev/documents.jsonl"
        if corpus_path.exists():
            corpus = [item["text"] for item in _corpus_documents(Path.cwd())]
    result = evaluate_sources([x for x in sources if x], corpus=corpus)
    result["generation_length"] = len(sources[0]) if sources else 0
    result["generation_length"] = generated_token_count
    result["generation_length_characters"] = len(sources[0]) if sources else 0
    result["deterministic_generation"] = deterministic_generation
    result["validation_cross_entropy"] = validation_cross_entropy
    result["perplexity"] = (
        perplexity(validation_cross_entropy) if validation_cross_entropy is not None else None
    )
    result["fim_exact_match"] = fim_exact
    result["fim_token_accuracy"] = fim_accuracy
    result["tokens_per_second"] = generation_tokens_per_second
    result["evaluation_scope"] = (
        "generated checkpoint sample"
        if checkpoint.exists() and not args.source
        else "provided source"
    )
    if args.output:
        output_path = _safe_project_path(Path.cwd(), args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_completion_evaluate(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    tasks_path = Path(args.tasks)
    payload = load_checkpoint_payload(checkpoint)
    metadata = payload["metadata"]
    expected_tokenizer_hash = metadata.get("tokenizer_hash")
    if expected_tokenizer_hash and sha256_file(tokenizer_path) != expected_tokenizer_hash:
        raise ValueError("tokenizer hash does not match checkpoint metadata")
    raw_config = metadata.get("resolved_config", {}).get("model", {})
    runtime_fields = set(RuntimeModelConfig.__dataclass_fields__)
    model = Transformer(
        RuntimeModelConfig(
            **{
                key: value
                for key, value in raw_config.items()
                if key != "schema_version" and key in runtime_fields
            }
        )
    )
    model.load_state_dict(payload["model"])
    device = _resolve_training_device(args.device)
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    tasks = load_completion_tasks(tasks_path)
    result = evaluate_completion_tasks(
        tasks,
        model_completion_generator(
            model,
            tokenizer,
            device=device,
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        ),
    )
    result["decoding"] = {
        "temperature": 0.0,
        "repetition_penalty": float(args.repetition_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
    }
    result["checkpoint"] = str(checkpoint)
    result["tokenizer"] = str(tokenizer_path)
    result["task_fixture"] = str(tasks_path)
    if args.output:
        output = _safe_project_path(Path.cwd(), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_fim_evaluate(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    tasks_path = Path(args.tasks)
    payload = load_checkpoint_payload(checkpoint)
    metadata = payload["metadata"]
    expected_tokenizer_hash = metadata.get("tokenizer_hash")
    if expected_tokenizer_hash and sha256_file(tokenizer_path) != expected_tokenizer_hash:
        raise ValueError("tokenizer hash does not match checkpoint metadata")
    raw_config = metadata.get("resolved_config", {}).get("model", {})
    runtime_fields = set(RuntimeModelConfig.__dataclass_fields__)
    model = Transformer(
        RuntimeModelConfig(
            **{
                key: value
                for key, value in raw_config.items()
                if key != "schema_version" and key in runtime_fields
            }
        )
    )
    model.load_state_dict(payload["model"])
    device = _resolve_training_device(args.device)
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    tasks = load_fim_tasks(tasks_path)
    result = evaluate_fim_tasks(
        tasks,
        model_fim_generator(
            model,
            tokenizer,
            device=device,
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        ),
        lambda text: tokenizer.encode(text).ids,
    )
    result["decoding"] = {
        "temperature": 0.0,
        "repetition_penalty": float(args.repetition_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
    }
    result["checkpoint"] = str(checkpoint)
    result["tokenizer"] = str(tokenizer_path)
    result["task_fixture"] = str(tasks_path)
    if args.output:
        output = _safe_project_path(Path.cwd(), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_model_audit(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    raw = _read_yaml(config_path, _MODEL_KEYS)
    if raw.get("planning_only"):
        report = {
            "schema_version": 1,
            "planning_only": True,
            "config_path": str(config_path),
            "declared": raw,
        }
    else:
        runtime_fields = set(RuntimeModelConfig.__dataclass_fields__)
        config = RuntimeModelConfig(
            **{key: value for key, value in raw.items() if key in runtime_fields}
        )
        report = model_audit(config)
        report["config_path"] = str(config_path)
        report["declared_training_tokens"] = raw.get("training_tokens")
    if args.output:
        output = _safe_project_path(Path.cwd(), args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _read_yaml(path: str | Path, allowed_keys: set[str] | None = None) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    if allowed_keys is not None:
        unknown = set(value) - allowed_keys
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    for key, allowed_nested in _NESTED_KEYS.items():
        if key not in value:
            continue
        nested = value[key]
        if key == "sources":
            if not isinstance(nested, list):
                raise ValueError("sources must be a list")
            for source in nested:
                if not isinstance(source, dict):
                    raise ValueError("each source must be a mapping")
                unknown = set(source) - allowed_nested
                if unknown:
                    raise ValueError(f"unknown source keys: {sorted(unknown)}")
            continue
        if not isinstance(nested, dict):
            raise ValueError(f"{key} must be a mapping")
        unknown = set(nested) - allowed_nested
        if unknown:
            raise ValueError(f"unknown {key} keys: {sorted(unknown)}")
    return value


def _safe_project_path(root: Path, value: str | Path) -> Path:
    """Resolve configured project paths without permitting traversal."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"absolute project paths are not allowed: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes project root: {value}")
    return resolved


def _validate_training_config(cfg: dict) -> None:
    sequence_length = int(cfg.get("sequence_length", 128))
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    objectives = cfg.get("objectives", {})
    causal = float(objectives.get("causal_fraction", 0.5))
    fim = float(objectives.get("fim_fraction", 0.5))
    if not 0 <= causal <= 1 or not 0 <= fim <= 1 or abs(causal + fim - 1) > 1e-6:
        raise ValueError("causal_fraction and fim_fraction must be probabilities summing to 1")
    if cfg.get("precision", "fp32") not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if bool(cfg.get("packed_streaming", False)) and not bool(cfg.get("streaming", False)):
        raise ValueError("packed_streaming requires streaming: true")
    for key in (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_tokens",
        "validation_interval_tokens",
        "validation_max_tokens",
        "checkpoint_interval_tokens",
        "sample_interval_tokens",
        "streaming_tokens_per_step_estimate",
    ):
        if key in cfg and int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    scheduler = cfg.get("scheduler", {})
    if int(scheduler.get("warmup_tokens", 0)) < 0:
        raise ValueError("warmup_tokens must not be negative")
    minimum_ratio = float(scheduler.get("minimum_learning_rate_ratio", 0.1))
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")
    validate_distributed_config(cfg.get("distributed"))


def _bounded_validation_batches(
    batches: Iterable[Mapping[str, torch.Tensor]], max_tokens: int | None
) -> Iterable[Mapping[str, torch.Tensor]]:
    """Bound validation work by actual input tokens, not source-record count.

    The approved corpus contains many short files.  Evaluating every validation
    record at every checkpoint can therefore perform tens of thousands of
    mostly-padding forward passes.  A deterministic token budget keeps
    periodic and final validation predictable while still weighting the loss by
    each batch's supervised-token count inside ``Trainer.evaluate_batches``.
    """
    if max_tokens is None:
        yield from batches
        return
    if max_tokens <= 0:
        raise ValueError("validation_max_tokens must be positive")
    consumed = 0
    for batch in batches:
        yield batch
        consumed += input_token_count(dict(batch))
        if consumed >= max_tokens:
            break


def _validation_max_tokens(cfg: Mapping[str, object]) -> int | None:
    """Resolve the bounded validation budget (default 32K actual tokens)."""
    value = cfg.get("validation_max_tokens", 32_768)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("validation_max_tokens must be an integer")
    return int(value)


def _validate_resume_identity(
    checkpoint_path: Path,
    model_config: dict,
    training_config: dict,
    tokenizer_hash: str,
    manifest_hash: str,
) -> None:
    """Fail closed before applying state from a mismatched training run."""
    metadata = load_checkpoint_payload(checkpoint_path)["metadata"]
    if metadata["tokenizer_hash"] != tokenizer_hash:
        raise ValueError("resume checkpoint tokenizer hash does not match current tokenizer")
    if metadata["manifest_hash"] != manifest_hash:
        raise ValueError("resume checkpoint manifest hash does not match current manifest")
    previous = metadata["resolved_config"]
    if previous.get("model") != model_config:
        raise ValueError("resume checkpoint model architecture does not match current model")
    previous_training = previous.get("training", {})
    critical = (
        "corpus_artifact",
        "manifest",
        "tokenizer_dir",
        "seed",
        "sequence_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "streaming_tokens_per_step_estimate",
        "objectives",
        "optimizer",
        "scheduler",
        "streaming",
        "packed_streaming",
        "exclude_compiler_harness",
        "shard_manifest",
    )
    mismatched = [key for key in critical if previous_training.get(key) != training_config.get(key)]
    if mismatched:
        raise ValueError(f"resume checkpoint data-critical config mismatch: {mismatched}")


def _cmd_corpus(args: argparse.Namespace) -> int:
    cfg = _read_yaml(args.config, _DATA_KEYS)
    for key in ("max_file_bytes", "max_files", "max_total_bytes", "max_path_depth"):
        if int(cfg.get(key, 1)) <= 0:
            raise ValueError(f"{key} must be positive")
    splits = cfg.get("splits", {"train": 0.9, "validation": 0.05, "test": 0.05})
    if (
        set(splits) != {"train", "validation", "test"}
        or any(not 0 <= float(value) <= 1 for value in splits.values())
        or abs(sum(float(value) for value in splits.values()) - 1) > 1e-6
    ):
        raise ValueError("splits must contain train/validation/test probabilities summing to 1")
    root = Path.cwd()
    require_approval = bool(cfg.get("require_approval", False))
    approval_path = None
    approval_records = []
    if cfg.get("approval_manifest"):
        approval_path = _safe_project_path(root, cfg["approval_manifest"])
        expected_approval_hash = cfg.get("approval_manifest_sha256")
        if expected_approval_hash and sha256_file(approval_path) != expected_approval_hash:
            raise ValueError("approval manifest hash does not match the configured digest")
        approval_records = load_approval_manifest(approval_path)
    elif require_approval:
        raise ValueError("require_approval requires approval_manifest")
    configured_licenses = frozenset(cfg.get("accepted_licenses", DEFAULT_ACCEPTED))
    unauthorized_licenses = configured_licenses - DEFAULT_ACCEPTED
    if unauthorized_licenses:
        raise ValueError(
            "accepted_licenses contains policy-disallowed values: "
            + ", ".join(sorted(unauthorized_licenses))
        )
    near_duplicate_threshold = float(cfg.get("near_duplicate_threshold", 0.85))
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    all_records: list[dict] = []
    all_contents: dict[str, str] = {}
    requested_sources = set(getattr(args, "source_id", None) or [])
    configured_sources = {str(item.get("source_id", "")) for item in cfg.get("sources", [])}
    unknown_sources = requested_sources - configured_sources
    if unknown_sources:
        raise ValueError(f"source ids are not configured: {sorted(unknown_sources)}")
    selected_config_sources = [
        source
        for source in cfg.get("sources", [])
        if not requested_sources or str(source.get("source_id", "")) in requested_sources
    ]
    real_source_intake = require_approval and any(
        str(source.get("commit_sha", "fixture")) != "fixture" for source in selected_config_sources
    )
    for source in cfg.get("sources", []):
        if requested_sources and str(source.get("source_id", "")) not in requested_sources:
            continue
        path = _safe_project_path(root, source["path"])
        repositories = _source_repositories(path)
        for repository in repositories:
            raw_source_uri = str(source.get("source_uri", f"fixture://{repository.name}"))
            source_uri = sanitize_source_uri(raw_source_uri)
            commit_sha = str(source.get("commit_sha", "fixture"))
            source_id = str(source.get("source_id", repository.name))
            approval = None
            receipt = None
            if require_approval:
                if source_uri != raw_source_uri:
                    raise ValueError(
                        "approved intake source_uri must not contain credentials, "
                        "queries, or fragments"
                    )
                approval = require_approved_source(
                    approval_records, source_id, source_uri, commit_sha
                )
                detected_license = detect_license(repository, configured_licenses)
                if (
                    detected_license.status != "accepted"
                    or detected_license.spdx != approval.license_spdx
                ):
                    raise ValueError(
                        "approved license does not match the source license detection: "
                        f"approved={approval.license_spdx}, detected={detected_license.spdx}"
                    )
                if commit_sha != "fixture":
                    if not source.get("receipt"):
                        raise ValueError("real approved sources require a snapshot receipt")
                    receipt_path = _safe_project_path(root, source["receipt"])
                    receipt = verify_snapshot_receipt(
                        snapshot_path=repository,
                        receipt_path=receipt_path,
                        expected_uri=source_uri,
                        expected_sha=commit_sha,
                    )
            records, contents = ingest_repository(
                repository,
                source_uri=source_uri,
                commit_sha=commit_sha,
                pipeline_version=str(cfg.get("pipeline_version", "working-tree")),
                seed=int(cfg.get("seed", 42)),
                retrieved_at=(
                    str(receipt["retrieved_at"])
                    if receipt is not None
                    else str(cfg.get("retrieved_at", "1970-01-01T00:00:00Z"))
                ),
                max_file_bytes=int(cfg.get("max_file_bytes", 1_000_000)),
                max_files=int(cfg.get("max_files", 10_000)),
                max_total_bytes=int(cfg.get("max_total_bytes", 100_000_000)),
                max_path_depth=int(cfg.get("max_path_depth", 32)),
                accepted_licenses=configured_licenses,
                approved_scope=approval.scope if approval is not None else None,
                near_duplicate_threshold=near_duplicate_threshold,
            )
            all_records.extend(records)
            all_contents.update(contents)
    # Re-run clustering across repository boundaries before splitting so exact
    # and near duplicates cannot leak between train/validation/test.
    assign_clusters(all_records, all_contents, near_duplicate_threshold)
    seen_normalized: set[str] = set()
    for record in sorted(all_records, key=lambda item: item["record_id"]):
        if not record["included"]:
            continue
        normalized = record["normalized_sha256"]
        if normalized in seen_normalized:
            record["included"] = False
            record["split"] = "excluded"
            record["exclusion_reasons"] = sorted(set(record["exclusion_reasons"]) | {"duplicate"})
        else:
            seen_normalized.add(normalized)
    assign_splits(
        all_records,
        int(cfg.get("seed", 42)),
        float(splits["train"]),
        float(splits["validation"]),
        group_by_repository=real_source_intake,
    )
    dry_run = bool(getattr(args, "dry_run", False))
    manifest = _safe_project_path(root, cfg.get("output_manifest", "manifests/dev.jsonl"))
    corpus_dir = _safe_project_path(root, cfg.get("output_corpus", "artifacts/corpus/dev"))
    if dry_run:
        digest = None
    else:
        append_manifest = bool(cfg.get("append_manifest", True))
        if manifest.exists() and not append_manifest and not bool(getattr(args, "replace", False)):
            raise ValueError(
                "output manifest already exists; choose a versioned path or pass --replace"
            )
        corpus_dir.mkdir(parents=True, exist_ok=True)
        documents_path = corpus_dir / "documents.jsonl"
        documents_temporary = corpus_dir / ".documents.jsonl.tmp"
        try:
            with documents_temporary.open("w", encoding="utf-8") as handle:
                for record in sorted(all_records, key=lambda item: item["record_id"]):
                    if record["included"]:
                        handle.write(
                            json.dumps(
                                {"record": record, "text": all_contents[record["record_id"]]},
                                sort_keys=True,
                            )
                            + "\n"
                        )
            digest = write_manifest(manifest, all_records, append=append_manifest)
            documents_temporary.replace(documents_path)
        finally:
            documents_temporary.unlink(missing_ok=True)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    normalized_splits: dict[str, set[str]] = defaultdict(set)
    repository_splits: dict[str, set[str]] = defaultdict(set)
    cluster_sizes: Counter[str] = Counter()
    for record in all_records:
        cluster_sizes[record["dedup_cluster"]] += 1
        if record["included"]:
            cluster_splits[record["dedup_cluster"]].add(record["split"])
            normalized_splits[record["normalized_sha256"]].add(record["split"])
            repository_splits[record["repository_id"]].add(record["split"])
    summary = {
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "approval_manifest": str(approval_path) if approval_path else None,
        "approval_required": require_approval,
        "approved_sources": len(approval_records),
        "selected_sources": len({record["source_uri"] for record in all_records}),
        "dry_run": dry_run,
        "records": len(all_records),
        "included": sum(x["included"] for x in all_records),
        "rejected": sum(not x["included"] for x in all_records),
        "bytes": sum(int(record["size_bytes"]) for record in all_records),
        "licenses": dict(sorted(Counter(record["license_spdx"] for record in all_records).items())),
        "source_types": dict(
            sorted(Counter(record["source_type"] for record in all_records).items())
        ),
        "splits": dict(
            sorted(Counter(record["split"] for record in all_records if record["included"]).items())
        ),
        "secret_scan_statuses": dict(
            sorted(Counter(record["secret_scan_status"] for record in all_records).items())
        ),
        "rejection_reasons": dict(
            sorted(
                Counter(
                    reason for record in all_records for reason in record["exclusion_reasons"]
                ).items()
            )
        ),
        "deduplication": {
            "clusters": len(cluster_sizes),
            "multi_record_clusters": sum(size > 1 for size in cluster_sizes.values()),
            "largest_cluster": max(cluster_sizes.values(), default=0),
            "cross_split_cluster_overlap": sum(len(items) > 1 for items in cluster_splits.values()),
            "cross_split_normalized_overlap": sum(
                len(items) > 1 for items in normalized_splits.values()
            ),
            "cross_split_repository_overlap": sum(
                len(items) > 1 for items in repository_splits.values()
            ),
        },
        "repository_grouping_enforced": real_source_intake,
        "repositories_by_split": dict(
            sorted(
                Counter(
                    next(iter(group_splits)) if len(group_splits) == 1 else "mixed"
                    for group_splits in repository_splits.values()
                ).items()
            )
        ),
    }
    if not dry_run:
        (corpus_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_approval_validate(args: argparse.Namespace) -> int:
    manifest = _safe_project_path(Path.cwd(), args.manifest)
    records = load_approval_manifest(manifest)
    statuses = {
        status: sum(record.status == status for record in records)
        for status in (
            "approved",
            "review",
            "rejected",
            "removed",
        )
    }
    print(
        json.dumps(
            {"manifest": str(manifest), "records": len(records), "statuses": statuses},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_approval_add(args: argparse.Namespace) -> int:
    manifest = _safe_project_path(Path.cwd(), args.manifest)
    record = make_approval(
        source_id=args.source_id,
        source_uri=args.source_uri,
        commit_sha=args.commit_sha,
        license_spdx=args.license_spdx,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
        scope=args.scope,
        status=args.status,
        notes=args.notes,
        supersedes=args.supersedes,
    )
    append_approval(manifest, record)
    print(
        json.dumps(
            {
                "approval_id": record.approval_id,
                "manifest": str(manifest),
                "source_id": record.source_id,
                "status": record.status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    cfg = _read_yaml(args.config, _DATA_KEYS)
    root = Path.cwd()
    if not cfg.get("approval_manifest"):
        raise ValueError("acquisition requires approval_manifest")
    approval_path = _safe_project_path(root, cfg["approval_manifest"])
    expected_approval_hash = cfg.get("approval_manifest_sha256")
    if expected_approval_hash and sha256_file(approval_path) != expected_approval_hash:
        raise ValueError("approval manifest hash does not match the configured digest")
    approval_records = load_approval_manifest(approval_path)
    destination_root = _safe_project_path(root, cfg.get("acquisition_root", "artifacts/sources"))
    requested = set(args.source_id or [])
    configured = {str(source.get("source_id", "")) for source in cfg.get("sources", [])}
    unknown = requested - configured
    if unknown:
        raise ValueError(f"source ids are not configured: {sorted(unknown)}")
    receipts = []
    for source in cfg.get("sources", []):
        source_id = str(source["source_id"])
        if requested and source_id not in requested:
            continue
        source_uri = str(source["source_uri"])
        commit_sha = str(source["commit_sha"])
        approval = require_approved_source(approval_records, source_id, source_uri, commit_sha)
        if commit_sha == "fixture":
            raise ValueError("fixture sources are not acquired from the network")
        expected_snapshot = _safe_project_path(root, source["path"])
        expected_receipt = _safe_project_path(root, source["receipt"])
        planned_snapshot = destination_root / source_id / commit_sha
        planned_receipt = destination_root / source_id / f"{commit_sha}.receipt.json"
        if expected_snapshot != planned_snapshot or expected_receipt != planned_receipt:
            raise ValueError(
                f"configured snapshot paths do not match acquisition layout: {source_id}"
            )
        receipt = acquire_git_source(
            source_id=source_id,
            source_uri=source_uri,
            commit_sha=commit_sha,
            destination_root=destination_root,
            max_file_bytes=int(
                cfg.get("acquisition_max_file_bytes", cfg.get("max_file_bytes", 1_000_000))
            ),
            max_files=int(cfg.get("acquisition_max_files", cfg.get("max_files", 10_000))),
            max_total_bytes=int(
                cfg.get(
                    "acquisition_max_total_bytes",
                    cfg.get("max_total_bytes", 100_000_000),
                )
            ),
            max_path_depth=int(
                cfg.get("acquisition_max_path_depth", cfg.get("max_path_depth", 32))
            ),
            timeout_seconds=int(cfg.get("acquisition_timeout_seconds", 120)),
            dry_run=bool(args.dry_run),
        )
        receipt["approved_license_spdx"] = approval.license_spdx
        receipts.append(receipt)
    print(
        json.dumps(
            {
                "approval_manifest": str(approval_path),
                "dry_run": bool(args.dry_run),
                "sources": receipts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _source_repositories(path: Path) -> list[Path]:
    """Accept either one repository snapshot or a directory of snapshots."""
    if not path.is_dir():
        return [path]
    repository_markers = {"LICENSE", "COPYING", "package.json", "tsconfig.json"}
    if any((path / marker).exists() for marker in repository_markers):
        return [path]
    return [candidate for candidate in sorted(path.iterdir()) if candidate.is_dir()]


def _corpus_documents(root: Path, path: str = "artifacts/corpus/dev/documents.jsonl") -> list[dict]:
    target = root / path
    if not target.exists():
        raise FileNotFoundError(f"corpus artifact missing: {target}; run fixture-corpus first")
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]


def _stream_shard_manifest(
    root: Path,
    corpus_path: Path,
    manifest_path: Path,
    *,
    tokenizer_hash: str,
    source_manifest_hash: str,
) -> ShardManifest:
    """Load or atomically create the one-shard JSONL stream index.

    The index is deliberately derived from the immutable corpus artifact and
    source manifest.  A stale index is rejected rather than silently changing
    the resume identity of a run.
    """
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ShardManifest.from_mapping(value)
        if (
            manifest.tokenizer_hash != tokenizer_hash
            or manifest.source_manifest_hash != source_manifest_hash
        ):
            raise ValueError("shard manifest hashes do not match current corpus/tokenizer")
        if len(manifest.shards) != 1 or manifest.shards[0].path != corpus_path.name:
            raise ValueError("streaming shard manifest does not identify the corpus artifact")
        return manifest
    if not corpus_path.is_file() or corpus_path.is_symlink():
        raise ValueError("streaming corpus artifact must be a regular file")
    records = 0
    with corpus_path.open("rb") as handle:
        for _ in handle:
            records += 1
    descriptor = ShardDescriptor(
        shard_id=corpus_path.name,
        path=corpus_path.name,
        records=records,
        sha256=sha256_file(corpus_path),
    )
    manifest = ShardManifest((descriptor,), tokenizer_hash, source_manifest_hash)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest


def _cmd_tokenizer(args: argparse.Namespace) -> int:
    cfg = _read_yaml(args.config, _TOKENIZER_KEYS)
    documents = _corpus_documents(
        Path.cwd(),
        _safe_project_path(
            Path.cwd(), cfg.get("corpus_artifact", "artifacts/corpus/dev/documents.jsonl")
        ),
    )
    output = _safe_project_path(Path.cwd(), cfg.get("output_dir", "artifacts/tokenizers/dev"))
    output.mkdir(parents=True, exist_ok=True)
    texts = [item["text"] for item in documents]
    metadata = train_tokenizer(
        texts,
        output / "tokenizer.json",
        vocab_size=int(cfg.get("vocab_size", 4096)),
        min_frequency=int(cfg.get("min_frequency", 2)),
    )
    metadata["metrics"] = evaluate_tokenizer(load_tokenizer(output / "tokenizer.json"), texts)
    (output / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _resolve_training_device(requested: str | None) -> torch.device:
    """Resolve and validate the configured training device before model move."""
    requested_device = str(requested or "cpu")
    try:
        resolved_device = torch.device(requested_device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid training device: {requested_device!r}") from exc
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but no CUDA device is available")
    return resolved_device


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = _read_yaml(args.config, _TRAINING_KEYS)
    _validate_training_config(cfg)
    distributed = validate_distributed_config(cfg.get("distributed"))
    if distributed.strategy != "single":
        raise ValueError(
            "distributed strategy is a planning contract; the milestone-1 CLI "
            "does not initialize DDP/FSDP process groups yet"
        )
    root = Path.cwd()
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    model_cfg = _read_yaml(
        _safe_project_path(root, cfg.get("model_config", "configs/model/dev.yaml")), _MODEL_KEYS
    )
    if model_cfg.get("planning_only"):
        raise ValueError("planning-only model configurations cannot be trained")
    runtime_fields = set(RuntimeModelConfig.__dataclass_fields__)
    model = Transformer(
        RuntimeModelConfig(
            **{k: v for k, v in model_cfg.items() if k != "schema_version" and k in runtime_fields}
        )
    )
    tokenizer_path = (
        _safe_project_path(root, cfg.get("tokenizer_dir", "artifacts/tokenizers/dev"))
        / "tokenizer.json"
    )
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        model.config.vocab_size == vocab_size or print(
            "warning: tokenizer/model vocabulary differs", file=sys.stderr
        )
    else:
        tokenizer = None
    corpus_artifact = str(cfg.get("corpus_artifact", "artifacts/corpus/dev/documents.jsonl"))
    streaming = bool(cfg.get("streaming", False))
    packed_streaming = bool(cfg.get("packed_streaming", False))
    if packed_streaming and not streaming:
        raise ValueError("packed_streaming requires streaming: true")
    corpus_path = _safe_project_path(root, corpus_artifact)
    train_docs: list[dict] = []
    validation_docs: list[dict] = []
    batches: list[dict[str, torch.Tensor]] = []
    validation_batches: list[dict[str, torch.Tensor]] = []
    objectives = cfg.get("objectives", {})
    fim_fraction = float(objectives.get("fim_fraction", 0.5))
    fim_min_span = int(objectives.get("fim_min_span", 1))
    fim_max_span = int(objectives.get("fim_max_span", 128))
    objective_examples = {"causal": 0, "fim": 0}
    objective_tokens = {"causal": 0, "fim": 0}
    if not streaming:
        docs = _corpus_documents(root, corpus_artifact)
        train_docs = [item for item in docs if item["record"].get("split") == "train"]
        validation_docs = [item for item in docs if item["record"].get("split") == "validation"]
        for item in train_docs:
            source_text = item["text"]
            record_id = str(item["record"]["record_id"])
            selected_objective = "causal"
            if tokenizer and use_fim(record_id, seed, fim_fraction):
                try:
                    source_text = make_fim(
                        source_text,
                        record_id,
                        seed,
                        min_span=fim_min_span,
                        max_span=fim_max_span,
                    ).serialized
                    selected_objective = "fim"
                except ValueError:
                    # Files shorter than the configured minimum remain valid causal samples.
                    pass
            ids = (
                tokenizer.encode(source_text).ids
                if tokenizer
                else [b % model.config.vocab_size for b in source_text.encode()]
            )
            if len(ids) < 2:
                continue
            for example in causal_examples(ids, int(cfg.get("sequence_length", 128)), 0):
                batches.append({k: torch.tensor([v], dtype=torch.long) for k, v in example.items()})
                objective_examples[selected_objective] += 1
                # ``Transformer`` predicts labels from position 1 onward;
                # position 0 is conditioning context, not a target.
                objective_tokens[selected_objective] += sum(example["loss_mask"][1:])
        for item in validation_docs:
            source_text = item["text"]
            ids = (
                tokenizer.encode(source_text).ids
                if tokenizer
                else [b % model.config.vocab_size for b in source_text.encode()]
            )
            if len(ids) >= 2:
                for example in causal_examples(ids, int(cfg.get("sequence_length", 128)), 0):
                    validation_batches.append(
                        {k: torch.tensor([v], dtype=torch.long) for k, v in example.items()}
                    )
        if not batches:
            raise ValueError("no trainable fixture documents")
    micro_batch_size = int(cfg.get("micro_batch_size", 1))
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    def combine_microbatches(
        examples: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        return [
            {
                key: torch.cat(
                    [example[key] for example in examples[start : start + micro_batch_size]]
                )
                for key in examples[start]
            }
            for start in range(0, len(examples), micro_batch_size)
        ]

    if not streaming:
        batches = combine_microbatches(batches)
        validation_batches = combine_microbatches(validation_batches)
    optimizer_cfg = cfg.get("optimizer", {})
    raw_betas = optimizer_cfg.get("betas", [0.9, 0.95])
    optimizer = build_adamw(
        model,
        float(optimizer_cfg.get("learning_rate", 1e-3)),
        float(optimizer_cfg.get("weight_decay", 0.1)),
        (float(raw_betas[0]), float(raw_betas[1])),
    )
    max_tokens = int(cfg.get("max_tokens", 32_768))
    sequence_length = int(cfg.get("sequence_length", 128))
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    if streaming:
        # Streaming examples are padded to ``sequence_length`` but may contain
        # only a few hundred real tokens.  Use an explicit conservative estimate
        # for the scheduler/step ceiling; actual attention-mask counts remain
        # authoritative for the token budget and reported progress.
        configured_estimate = cfg.get("streaming_tokens_per_step_estimate", 300)
        if isinstance(configured_estimate, bool) or not isinstance(
            configured_estimate, (int, float, str)
        ):
            raise ValueError("streaming_tokens_per_step_estimate must be an integer")
        estimated_tokens_per_step = int(configured_estimate)
        if estimated_tokens_per_step <= 0:
            raise ValueError("streaming_tokens_per_step_estimate must be positive")
    else:
        estimated_tokens_per_step = sequence_length * micro_batch_size * accumulation
    estimated_steps = max(2, math.ceil(max_tokens / estimated_tokens_per_step))
    scheduler_cfg = cfg.get("scheduler", {})
    warmup_tokens = int(scheduler_cfg.get("warmup_tokens", estimated_tokens_per_step))
    warmup_steps = min(estimated_steps - 1, math.ceil(warmup_tokens / estimated_tokens_per_step))
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_steps,
        estimated_steps,
        float(scheduler_cfg.get("minimum_learning_rate_ratio", 0.1)),
    )
    run_name = str(cfg.get("run_name", "dev"))
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("run_name must be a single safe path component")
    run = create_run_dir(
        _safe_project_path(root, cfg.get("output_root", "artifacts/runs")), run_name, cfg, seed
    )
    resolved_device = _resolve_training_device(cfg.get("device"))
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        device=resolved_device,
        gradient_accumulation_steps=accumulation,
        max_grad_norm=float(cfg.get("gradient_clip_norm", 1.0)),
        metrics_path=run / "metrics.jsonl",
        precision=str(cfg.get("precision", "fp32")),
    )
    manifest_path = _safe_project_path(root, cfg.get("manifest", "manifests/dev.jsonl"))
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path.exists() else "unavailable"
    manifest_hash = sha256_file(manifest_path) if manifest_path.exists() else "unavailable"
    resolved_config = {"model": model_cfg, "training": cfg}
    if cfg.get("resume_from"):
        resume_path = _safe_project_path(root, cfg["resume_from"])
        _validate_resume_identity(resume_path, model_cfg, cfg, tokenizer_hash, manifest_hash)
        trainer.resume(resume_path)

    stream_dataset: TokenizedStreamingDataset | None = None
    validation_stream_dataset: TokenizedStreamingDataset | None = None
    stream_batcher: TokenizedStreamingBatcher | PackedTokenBlockBatcher | None = None
    if streaming:
        if tokenizer is None:
            raise ValueError("streaming training requires a tokenizer artifact")
        shard_manifest_path = _safe_project_path(
            root,
            cfg.get("shard_manifest", str(corpus_path.parent / "shards.json")),
        )
        shard_manifest = _stream_shard_manifest(
            root,
            corpus_path,
            shard_manifest_path,
            tokenizer_hash=tokenizer_hash,
            source_manifest_hash=manifest_hash,
        )
        shard_dataset = StreamingShardDataset(
            shard_manifest,
            corpus_path.parent,
            max_line_bytes=int(cfg.get("max_line_bytes", 16 * 1024 * 1024)),
        )
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("streaming tokenizer is missing <pad>")
        stream_dataset = TokenizedStreamingDataset(
            shard_dataset,
            tokenizer,
            context_length=sequence_length,
            pad_id=pad_id,
            seed=seed,
            fim_fraction=fim_fraction,
            fim_min_span=fim_min_span,
            fim_max_span=fim_max_span,
            split="train",
            tokenizer_hash=tokenizer_hash,
            exclude_compiler_harness=bool(cfg.get("exclude_compiler_harness", False)),
        )
        validation_stream_dataset = TokenizedStreamingDataset(
            shard_dataset,
            tokenizer,
            context_length=sequence_length,
            pad_id=pad_id,
            seed=seed,
            fim_fraction=0.0,
            fim_min_span=fim_min_span,
            fim_max_span=fim_max_span,
            split="validation",
            tokenizer_hash=tokenizer_hash,
            exclude_compiler_harness=bool(cfg.get("exclude_compiler_harness", False)),
        )
        resume_cursor = (
            DataCursor.from_mapping(trainer.data_position)
            if trainer.data_position is not None
            else stream_dataset.initial_cursor()
        )
        if packed_streaming:
            eos_id = tokenizer.token_to_id("<eos>")
            if eos_id is None:
                raise ValueError("packed streaming requires an <eos> tokenizer token")
            stream_batcher = PackedTokenBlockBatcher(
                stream_dataset,
                eos_id=eos_id,
                batch_size=micro_batch_size,
                cursor=resume_cursor,
                epochs=None,
            )
        else:
            stream_batcher = TokenizedStreamingBatcher(
                stream_dataset,
                batch_size=micro_batch_size,
                cursor=resume_cursor,
                epochs=None,
            )

    def save_progress(state) -> None:
        trainer.save(
            run / "checkpoints" / f"tokens-{state.tokens_processed}",
            resolved_config,
            tokenizer_hash,
            manifest_hash,
            "unavailable",
        )

    def save_interrupt(state) -> None:
        trainer.save(
            run / "checkpoints" / "interrupted",
            resolved_config,
            tokenizer_hash,
            manifest_hash,
            "unavailable",
        )

    def write_sample(state) -> None:
        prompt_text = (
            train_docs[0]["text"][:128]
            if train_docs
            else "export function generatedCompletion(input: string): string {\n  return "
        )
        prompt_ids = (
            tokenizer.encode(prompt_text).ids
            if tokenizer
            else [byte % model.config.vocab_size for byte in prompt_text.encode("utf-8")]
        )
        prompt_ids = prompt_ids[-max(1, model.config.context_length - 16) :]
        generated = generate(
            model,
            torch.tensor([prompt_ids], dtype=torch.long, device=trainer.device),
            min(16, model.config.context_length - len(prompt_ids)),
        )[0].tolist()
        model.train()
        text = (
            tokenizer.decode(generated) if tokenizer else bytes(generated).decode(errors="replace")
        )
        target = run / "samples" / f"tokens-{state.tokens_processed}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"tokens_processed": state.tokens_processed, "text": text}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def write_validation(state) -> None:
        validation_batches_iter: Iterable[Mapping[str, torch.Tensor]]
        if validation_stream_dataset is not None:
            if packed_streaming:
                eos_id = tokenizer.token_to_id("<eos>") if tokenizer else None
                if eos_id is None:
                    raise ValueError("packed validation requires an <eos> tokenizer token")
                validation_batches_iter = PackedTokenBlockBatcher(
                    validation_stream_dataset,
                    eos_id=eos_id,
                    batch_size=micro_batch_size,
                    epochs=1,
                )
            else:
                validation_batches_iter = TokenizedStreamingBatcher(
                    validation_stream_dataset,
                    batch_size=micro_batch_size,
                    epochs=1,
                )
        else:
            validation_batches_iter = validation_batches
        loss = trainer.evaluate_batches(
            _bounded_validation_batches(validation_batches_iter, _validation_max_tokens(cfg))
        )
        if loss is not None:
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"type": "validation", "step": state.global_step, "loss": loss},
                        sort_keys=True,
                    )
                    + "\n"
                )

    # ``estimated_steps`` is a scheduler horizon, not a hard stop for a
    # record-oriented stream.  Short files can deliver far fewer real tokens
    # than the configured estimate; allow up to one optimizer step per target
    # token as a conservative loop ceiling and let ``max_tokens`` remain the
    # authoritative stop condition.
    max_train_steps = estimated_steps
    if streaming:
        max_train_steps = max(estimated_steps, max_tokens)
    try:
        batch_stream: Iterable[Mapping[str, torch.Tensor]]
        if stream_batcher is not None:
            batch_stream = stream_batcher
        else:
            batch_stream = islice(cycle(batches), trainer.state.data_cursor % len(batches), None)
        metrics = trainer.train_steps(
            batch_stream,
            max_steps=trainer.state.global_step + max_train_steps,
            max_tokens=max_tokens,
            checkpoint_interval_tokens=int(cfg.get("checkpoint_interval_tokens", max_tokens)),
            sample_interval_tokens=int(cfg.get("sample_interval_tokens", max_tokens)),
            validation_interval_tokens=int(cfg.get("validation_interval_tokens", max_tokens)),
            checkpoint_callback=save_progress,
            sample_callback=write_sample,
            validation_callback=write_validation,
            interrupt_callback=save_interrupt,
        )
    except KeyboardInterrupt:
        interrupted_path = run / "checkpoints" / "interrupted"
        print(
            f"training interrupted; resumable checkpoint saved at {interrupted_path}",
            file=sys.stderr,
        )
        return 130
    if validation_stream_dataset is not None:
        final_validation_batches: Iterable[Mapping[str, torch.Tensor]] = TokenizedStreamingBatcher(
            validation_stream_dataset,
            batch_size=micro_batch_size,
            epochs=1,
        )
    else:
        final_validation_batches = validation_batches
    validation_loss = trainer.evaluate_batches(
        _bounded_validation_batches(final_validation_batches, _validation_max_tokens(cfg))
    )
    if validation_loss is not None:
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "validation",
                        "step": trainer.state.global_step,
                        "loss": validation_loss,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    checkpoint = trainer.save(
        run / "checkpoints" / "latest",
        resolved_config,
        tokenizer_hash,
        manifest_hash,
        "unavailable",
    )
    if tokenizer_path.exists():
        (run / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    if manifest_path.exists():
        (run / "data-manifest.json").write_bytes(manifest_path.read_bytes())
    report = {
        "run_id": cfg.get("run_name", "dev"),
        "parameter_count": count_parameters(model),
        "steps": len(metrics),
        "tokens_processed": trainer.state.tokens_processed,
        "checkpoint": str(checkpoint),
        "final_loss": metrics[-1].loss if metrics else None,
        "validation_loss": validation_loss,
        "objective_examples": (trainer.objective_examples if streaming else objective_examples),
        "objective_tokens": trainer.objective_tokens if streaming else objective_tokens,
        "seed": seed,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (run / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run / "report.md").write_text(
        "# Development run\n\n"
        f"- Steps: {report['steps']}\n"
        f"- Parameters: {report['parameter_count']}\n"
        f"- Final loss: {report['final_loss']}\n"
        f"- Checkpoint: `{checkpoint}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ts-coder")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--prompt-file")
    gen.add_argument("--prompt", default="")
    gen.add_argument("--max-new-tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--tokenizer")
    gen.set_defaults(func=_cmd_generate)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--source")
    evaluate.add_argument("--config")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=_cmd_evaluate)
    completion_evaluate = sub.add_parser("completion-evaluate")
    completion_evaluate.add_argument("--checkpoint", required=True)
    completion_evaluate.add_argument("--tokenizer", required=True)
    completion_evaluate.add_argument("--tasks", default="fixtures/evaluation/completion-tasks.json")
    completion_evaluate.add_argument("--output")
    completion_evaluate.add_argument("--device", default="cpu")
    completion_evaluate.add_argument("--repetition-penalty", type=float, default=1.0)
    completion_evaluate.add_argument("--no-repeat-ngram-size", type=int, default=0)
    completion_evaluate.set_defaults(func=_cmd_completion_evaluate)
    fim_evaluate = sub.add_parser("fim-evaluate")
    fim_evaluate.add_argument("--checkpoint", required=True)
    fim_evaluate.add_argument("--tokenizer", required=True)
    fim_evaluate.add_argument("--tasks", default="fixtures/evaluation/fim-tasks.json")
    fim_evaluate.add_argument("--output")
    fim_evaluate.add_argument("--device", default="cpu")
    fim_evaluate.add_argument("--repetition-penalty", type=float, default=1.0)
    fim_evaluate.add_argument("--no-repeat-ngram-size", type=int, default=0)
    fim_evaluate.set_defaults(func=_cmd_fim_evaluate)
    model_audit_parser = sub.add_parser("model-audit")
    model_audit_parser.add_argument("--config", required=True)
    model_audit_parser.add_argument("--output")
    model_audit_parser.set_defaults(func=_cmd_model_audit)
    corpus = sub.add_parser("corpus")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_sub.add_parser("build")
    corpus_build.add_argument("--config", required=True)
    corpus_build.add_argument("--source-id", action="append")
    corpus_build.add_argument(
        "--replace", action="store_true", help="replace an existing non-append manifest"
    )
    corpus_build.add_argument(
        "--dry-run", action="store_true", help="validate and summarize without writing artifacts"
    )
    corpus_build.set_defaults(func=_cmd_corpus)
    corpus_acquire = corpus_sub.add_parser("acquire")
    corpus_acquire.add_argument("--config", required=True)
    corpus_acquire.add_argument("--source-id", action="append")
    corpus_acquire.add_argument(
        "--dry-run",
        action="store_true",
        help="validate approvals and paths without network or writes",
    )
    corpus_acquire.set_defaults(func=_cmd_acquire)
    approvals = corpus_sub.add_parser("approvals")
    approvals_sub = approvals.add_subparsers(dest="approval_command", required=True)
    approvals_validate = approvals_sub.add_parser("validate")
    approvals_validate.add_argument("--manifest", required=True)
    approvals_validate.set_defaults(func=_cmd_approval_validate)
    approvals_add = approvals_sub.add_parser("add")
    approvals_add.add_argument("--manifest", required=True)
    approvals_add.add_argument("--source-id", required=True)
    approvals_add.add_argument("--source-uri", required=True)
    approvals_add.add_argument("--commit-sha", required=True)
    approvals_add.add_argument("--license-spdx", required=True)
    approvals_add.add_argument("--approved-by", required=True)
    approvals_add.add_argument("--approved-at", required=True)
    approvals_add.add_argument("--scope", action="append", required=True)
    approvals_add.add_argument(
        "--status", choices=("approved", "review", "rejected", "removed"), default="approved"
    )
    approvals_add.add_argument("--notes", default="")
    approvals_add.add_argument("--supersedes")
    approvals_add.set_defaults(func=_cmd_approval_add)
    tokenizer = sub.add_parser("tokenizer")
    tokenizer_sub = tokenizer.add_subparsers(dest="tokenizer_command", required=True)
    tokenizer_train = tokenizer_sub.add_parser("train")
    tokenizer_train.add_argument("--config", required=True)
    tokenizer_train.set_defaults(func=_cmd_tokenizer)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.set_defaults(func=_cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
