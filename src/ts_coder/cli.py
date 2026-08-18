"""Hardened command-line entry point.

The original implementation is retained in :mod:`ts_coder.cli_legacy` so the
large vertical slice remains reviewable. This module re-exports its helpers for
compatibility and replaces commands that cross artifact, evaluation, or approval
boundaries with fail-closed implementations.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from . import cli_legacy as _legacy
from .artifacts import (
    LoadedModelBundle,
    load_model_bundle,
    verify_streaming_data_bundle,
)
from .corpus.approval import load_approval_manifest
from .data.causal import causal_examples
from .data.streaming import ShardManifest, StreamingShardDataset
from .data.token_stream import shard_manifest_hash
from .evaluation.completion import (
    evaluate_completion_tasks,
    load_completion_tasks,
    model_completion_generator,
)
from .evaluation.fim_completion import (
    evaluate_fim_tasks,
    load_fim_tasks,
    model_fim_generator,
)
from .evaluation.loss import perplexity
from .evaluation.runner import evaluate_sources, peak_memory_bytes
from .model.generation import generate
from .reproducibility import seed_everything, sha256_file
from .tokenizer.metrics import evaluate_tokenizer
from .tokenizer.special_tokens import SPECIAL_TOKENS
from .tokenizer.trainer import load_tokenizer, train_tokenizer
from .training.authorization import require_training_authorization
from .training.metrics import input_token_count, supervised_token_count

# Preserve helpers imported by existing tests and internal callers. Hardened
# definitions below intentionally replace selected names.
for _export_name in dir(_legacy):
    if not _export_name.startswith("__"):
        globals()[_export_name] = getattr(_legacy, _export_name)

_EXPLICIT_TRAINING_AUTHORIZATION_THRESHOLD = 1_000_000
_SUPPORTED_EVALUATION_METRICS = {
    "cross_entropy",
    "perplexity",
    "fim_exact_match",
    "fim_token_accuracy",
    "syntax_parse_rate",
    "compilation_rate",
    "diagnostic_count",
    "generation_length",
    "repetition_rate",
    "exact_training_match_rate",
    "longest_matching_span",
    "tokens_per_second",
    "peak_memory",
    "deterministic_generation",
    "security_clean_rate",
}
_EVALUATION_KEYS = {
    "schema_version",
    "seed",
    "manifest",
    "split",
    "max_new_tokens",
    "max_evaluation_tokens",
    "generation",
    "metrics",
    "typescript_tool",
    "compile_timeout_seconds",
    "completion_tasks",
    "fim_tasks",
    "output",
}
_GENERATION_KEYS = {"temperature", "top_k", "top_p", "use_kv_cache"}


def _read_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"configuration path must be a regular file: {source}")
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _read_evaluation_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = _read_mapping(path)
    unknown = set(value) - _EVALUATION_KEYS
    if unknown:
        raise ValueError(f"unknown evaluation configuration keys: {sorted(unknown)}")
    generation = value.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping")
    unknown_generation = set(generation) - _GENERATION_KEYS
    if unknown_generation:
        raise ValueError(f"unknown generation keys: {sorted(unknown_generation)}")
    metrics = value.get("metrics", sorted(_SUPPORTED_EVALUATION_METRICS))
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise ValueError("metrics must be a list of metric names")
    unsupported = set(metrics) - _SUPPORTED_EVALUATION_METRICS
    if unsupported:
        raise ValueError(f"unsupported evaluation metrics: {sorted(unsupported)}")
    return value


def _regular_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"path must be a regular file: {path}")
    return path.resolve()


def _safe_regular_project_path(root: Path, value: str | Path) -> Path:
    path = _legacy._safe_project_path(root, value)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"project artifact must be a regular file: {path}")
    return path


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_documents(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"corpus artifact must be a regular file: {path}")
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("record"), dict)
            or not isinstance(value.get("text"), str)
        ):
            raise ValueError(f"invalid corpus document at line {line_number}")
        documents.append(value)
    return documents


def _typescript_tool_root(root: Path, cfg: Mapping[str, Any]) -> Path:
    value = cfg.get("typescript_tool", "tools/typescript")
    path = _legacy._safe_project_path(root, str(value))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"TypeScript tool root must be a regular directory: {path}")
    return path


def _generation_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = cfg.get("generation", {})
    generation = raw if isinstance(raw, Mapping) else {}
    temperature = float(generation.get("temperature", 0.0))
    top_k = int(generation.get("top_k", 0))
    top_p = float(generation.get("top_p", 1.0))
    use_kv_cache = generation.get("use_kv_cache", True)
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid evaluation generation settings")
    if not isinstance(use_kv_cache, bool):
        raise ValueError("use_kv_cache must be a boolean")
    return {
        "temperature": temperature,
        "top_k": top_k or None,
        "top_p": top_p,
        "use_kv_cache": use_kv_cache,
    }


def _verify_checkpoint_corpus(bundle: LoadedModelBundle, root: Path, corpus_path: Path) -> None:
    training_cfg = bundle.training_config
    if not bool(training_cfg.get("streaming", False)):
        return
    manifest_value = training_cfg.get("shard_manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("streaming checkpoint lacks a shard_manifest path")
    shard_path = _safe_regular_project_path(root, manifest_value)
    raw = json.loads(shard_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("shard manifest root must be an object")
    manifest = ShardManifest.from_mapping(raw)
    expected_hash = bundle.metadata.get("shard_manifest_hash")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("streaming checkpoint lacks a shard manifest identity")
    if shard_manifest_hash(manifest) != expected_hash:
        raise ValueError("shard manifest does not match checkpoint metadata")
    if manifest.tokenizer_hash != bundle.metadata.get("tokenizer_hash"):
        raise ValueError("shard manifest tokenizer hash does not match checkpoint metadata")
    if manifest.source_manifest_hash != bundle.metadata.get("manifest_hash"):
        raise ValueError("shard manifest source hash does not match checkpoint metadata")
    if len(manifest.shards) != 1:
        raise ValueError("current training contract requires one corpus JSONL shard")
    described = (corpus_path.parent / manifest.shards[0].path).resolve()
    if described != corpus_path.resolve():
        raise ValueError("shard manifest does not identify the checkpoint corpus artifact")
    StreamingShardDataset(manifest, corpus_path.parent, verify_shards=True)


def _weighted_cross_entropy(
    bundle: LoadedModelBundle,
    documents: list[dict[str, Any]],
    max_input_tokens: int,
) -> float | None:
    if max_input_tokens <= 0:
        raise ValueError("max_evaluation_tokens must be positive")
    pad_id = bundle.tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("tokenizer is missing <pad>")
    weighted_loss = 0.0
    target_tokens = 0
    input_tokens = 0
    with torch.no_grad():
        for item in documents:
            ids = bundle.tokenizer.encode(item["text"]).ids
            for example in causal_examples(ids, bundle.model.config.context_length, pad_id):
                batch = {
                    key: torch.tensor([value], dtype=torch.long, device=bundle.device)
                    for key, value in example.items()
                }
                output = bundle.model(**batch)
                count = supervised_token_count(batch)
                if output.loss is not None and torch.isfinite(output.loss) and count:
                    weighted_loss += float(output.loss) * count
                    target_tokens += count
                input_tokens += input_token_count(batch)
                if input_tokens >= max_input_tokens:
                    return weighted_loss / target_tokens if target_tokens else None
    return weighted_loss / target_tokens if target_tokens else None


def _diagnostic_count(result: Mapping[str, Any]) -> int:
    count = 0
    tasks = result.get("tasks", [])
    if not isinstance(tasks, list):
        return 0
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        for key in ("syntax", "compile"):
            value = task.get(key, {})
            if isinstance(value, Mapping) and isinstance(value.get("diagnostics"), list):
                count += len(value["diagnostics"])
    return count


def _selected_metrics(
    requested: list[str], values: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = {
        "cross_entropy": "validation_cross_entropy",
        "peak_memory": "peak_memory_bytes",
        "fim_exact_match": "fim_exact_match",
    }
    selected: dict[str, Any] = {}
    canonical: dict[str, Any] = {}
    for name in requested:
        value = values.get(name)
        selected[name] = value
        canonical[aliases.get(name, name)] = value
    return selected, canonical


def _cmd_generate(args) -> int:
    bundle = load_model_bundle(args.checkpoint, tokenizer=args.tokenizer)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt:
        raise ValueError("generation prompt must not be empty")
    seed_everything(int(args.seed))
    prompt_ids = bundle.tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("tokenizer produced an empty prompt")
    available = bundle.model.config.context_length - len(prompt_ids)
    if available <= 0:
        raise ValueError("generation prompt exceeds model context length")
    requested = min(int(args.max_new_tokens), available)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
    output = generate(
        bundle.model,
        ids,
        max_new_tokens=requested,
        temperature=float(args.temperature),
        top_k=int(args.top_k) or None,
        top_p=float(args.top_p),
        use_kv_cache=True,
    )
    new_ids = output[0, ids.shape[1] :].tolist()
    generated = bundle.tokenizer.decode(new_ids, skip_special_tokens=False)
    print(
        json.dumps(
            {
                "checkpoint": str(bundle.checkpoint_path),
                "tokenizer": str(bundle.tokenizer_path),
                "seed": int(args.seed),
                "tokens": len(new_ids),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    sys.stdout.write(generated)
    return 0


def _cmd_completion_evaluate(args) -> int:
    root = Path.cwd()
    bundle = load_model_bundle(
        args.checkpoint, tokenizer=args.tokenizer, device=getattr(args, "device", "cpu")
    )
    tasks_path = _regular_path(args.tasks)
    tool_root = _typescript_tool_root(root, {})
    result = evaluate_completion_tasks(
        load_completion_tasks(tasks_path),
        model_completion_generator(
            bundle.model,
            bundle.tokenizer,
            device=bundle.device,
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        ),
        tool_root=tool_root,
        compile_timeout_seconds=10,
    )
    result["decoding"] = {
        "temperature": 0.0,
        "repetition_penalty": float(args.repetition_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
        "use_kv_cache": True,
    }
    result["checkpoint"] = str(bundle.checkpoint_path)
    result["tokenizer"] = str(bundle.tokenizer_path)
    result["task_fixture"] = str(tasks_path)
    if args.output:
        _write_json_atomic(_legacy._safe_project_path(root, args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_fim_evaluate(args) -> int:
    root = Path.cwd()
    bundle = load_model_bundle(
        args.checkpoint, tokenizer=args.tokenizer, device=getattr(args, "device", "cpu")
    )
    tasks_path = _regular_path(args.tasks)
    tool_root = _typescript_tool_root(root, {})
    result = evaluate_fim_tasks(
        load_fim_tasks(tasks_path),
        model_fim_generator(
            bundle.model,
            bundle.tokenizer,
            device=bundle.device,
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        ),
        lambda text: bundle.tokenizer.encode(text).ids,
        tool_root=tool_root,
        compile_timeout_seconds=10,
    )
    result["decoding"] = {
        "temperature": 0.0,
        "repetition_penalty": float(args.repetition_penalty),
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
        "use_kv_cache": True,
    }
    result["checkpoint"] = str(bundle.checkpoint_path)
    result["tokenizer"] = str(bundle.tokenizer_path)
    result["task_fixture"] = str(tasks_path)
    if args.output:
        _write_json_atomic(_legacy._safe_project_path(root, args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_evaluate(args) -> int:
    root = Path.cwd()
    cfg = _read_evaluation_config(args.config)
    if args.output is None and cfg.get("output"):
        args.output = cfg["output"]
    requested = list(cfg.get("metrics", sorted(_SUPPORTED_EVALUATION_METRICS)))
    tool_root = _typescript_tool_root(root, cfg)
    compile_timeout = int(cfg.get("compile_timeout_seconds", 10))
    if compile_timeout <= 0:
        raise ValueError("compile_timeout_seconds must be positive")

    if args.source:
        source_path = _regular_path(args.source)
        source = source_path.read_text(encoding="utf-8")
        source_result = evaluate_sources(
            [source], tool_root=tool_root, compile_timeout_seconds=compile_timeout
        )
        values = {
            "cross_entropy": None,
            "perplexity": None,
            "fim_exact_match": None,
            "fim_token_accuracy": None,
            "syntax_parse_rate": source_result["syntax_parse_rate"],
            "compilation_rate": source_result["compilation_rate"],
            "diagnostic_count": source_result["diagnostic_count"],
            "generation_length": None,
            "repetition_rate": source_result["repetition_rate"],
            "exact_training_match_rate": None,
            "longest_matching_span": None,
            "tokens_per_second": None,
            "peak_memory": source_result["peak_memory_bytes"],
            "deterministic_generation": None,
            "security_clean_rate": source_result["security_clean_rate"],
        }
        selected, canonical = _selected_metrics(requested, values)
        result: dict[str, Any] = {
            "schema_version": 2,
            "evaluation_scope": "provided source",
            "source": str(source_path),
            "metrics": selected,
            **canonical,
        }
    else:
        checkpoint = args.checkpoint or "artifacts/runs/dev/checkpoints/latest"
        bundle = load_model_bundle(checkpoint)
        seed_everything(int(cfg.get("seed", 42)))
        training_cfg = bundle.training_config
        manifest_value = cfg.get("manifest", training_cfg.get("manifest"))
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError("evaluation requires a manifest path")
        manifest_path = _safe_regular_project_path(root, manifest_value)
        expected_manifest_hash = bundle.metadata.get("manifest_hash")
        if (
            not isinstance(expected_manifest_hash, str)
            or expected_manifest_hash == "unavailable"
            or sha256_file(manifest_path) != expected_manifest_hash
        ):
            raise ValueError("evaluation manifest hash does not match checkpoint metadata")
        shard_manifest_path: Path | None = None
        if training_cfg.get("streaming"):
            corpus_path, shard_manifest_path = verify_streaming_data_bundle(
                bundle, project_root=root
            )
        else:
            corpus_value = training_cfg.get("corpus_artifact")
            if not isinstance(corpus_value, str) or not corpus_value:
                raise ValueError("checkpoint training configuration lacks corpus_artifact")
            corpus_path = _safe_regular_project_path(root, corpus_value)
        _verify_checkpoint_corpus(bundle, root, corpus_path)
        documents = _load_documents(corpus_path)
        split = str(cfg.get("split", "test"))
        evaluation_documents = [item for item in documents if item["record"].get("split") == split]
        if not evaluation_documents:
            raise ValueError(f"corpus contains no documents in evaluation split {split!r}")
        training_corpus = [
            item["text"] for item in documents if item["record"].get("split") == "train"
        ]
        loss = _weighted_cross_entropy(
            bundle,
            evaluation_documents,
            int(cfg.get("max_evaluation_tokens", 32_768)),
        )
        completion_tasks_path = _safe_regular_project_path(
            root, cfg.get("completion_tasks", "fixtures/evaluation/completion-tasks.json")
        )
        fim_tasks_path = _safe_regular_project_path(
            root, cfg.get("fim_tasks", "fixtures/evaluation/fim-tasks.json")
        )
        generation_cfg = _generation_settings(cfg)
        max_new_tokens = int(cfg.get("max_new_tokens", 256))
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        completion_tasks = load_completion_tasks(completion_tasks_path)
        fim_tasks = load_fim_tasks(fim_tasks_path)
        for task in (*completion_tasks, *fim_tasks):
            task["max_new_tokens"] = min(int(task["max_new_tokens"]), max_new_tokens)
        started = time.perf_counter()
        completion_result = evaluate_completion_tasks(
            completion_tasks,
            model_completion_generator(
                bundle.model,
                bundle.tokenizer,
                device=bundle.device,
                use_kv_cache=generation_cfg["use_kv_cache"],
                temperature=generation_cfg["temperature"],
                top_k=generation_cfg["top_k"],
                top_p=generation_cfg["top_p"],
            ),
            tool_root=tool_root,
            compile_timeout_seconds=compile_timeout,
        )
        fim_result = evaluate_fim_tasks(
            fim_tasks,
            model_fim_generator(
                bundle.model,
                bundle.tokenizer,
                device=bundle.device,
                use_kv_cache=generation_cfg["use_kv_cache"],
                temperature=generation_cfg["temperature"],
                top_k=generation_cfg["top_k"],
                top_p=generation_cfg["top_p"],
            ),
            lambda text: bundle.tokenizer.encode(text).ids,
            tool_root=tool_root,
            compile_timeout_seconds=compile_timeout,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        held_out_sources = [task["source"] for task in completion_result["tasks"]]
        held_out_sources.extend(task["reconstructed"] for task in fim_result["tasks"])
        source_result = evaluate_sources(
            held_out_sources,
            corpus=training_corpus,
            tool_root=tool_root,
            compile_timeout_seconds=compile_timeout,
            compute_longest_matching_span="longest_matching_span" in requested,
        )
        completions = [task["completion"] for task in completion_result["tasks"]]
        generated_tokens = sum(len(bundle.tokenizer.encode(text).ids) for text in completions)
        generated_tokens += sum(
            len(bundle.tokenizer.encode(task["completion"]).ids) for task in fim_result["tasks"]
        )
        total_tasks = completion_result["task_count"] + fim_result["task_count"]
        diagnostic_count = _diagnostic_count(completion_result) + _diagnostic_count(fim_result)
        deterministic = (
            completion_result["deterministic_rate"] == 1.0
            and fim_result["deterministic_rate"] == 1.0
        )
        values = {
            "cross_entropy": loss,
            "perplexity": perplexity(loss) if loss is not None else None,
            "fim_exact_match": fim_result["fim_exact_match_rate"],
            "fim_token_accuracy": fim_result["fim_token_accuracy"],
            "syntax_parse_rate": source_result["syntax_parse_rate"],
            "compilation_rate": source_result["compilation_rate"],
            "diagnostic_count": diagnostic_count,
            "generation_length": generated_tokens / max(total_tasks, 1),
            "repetition_rate": (
                completion_result["mean_repetition_rate"] * completion_result["task_count"]
                + fim_result["mean_repetition_rate"] * fim_result["task_count"]
            )
            / max(total_tasks, 1),
            "exact_training_match_rate": source_result["exact_training_match_rate"],
            "longest_matching_span": source_result["longest_matching_span"],
            "tokens_per_second": generated_tokens / elapsed,
            "peak_memory": peak_memory_bytes(),
            "deterministic_generation": deterministic,
            "security_clean_rate": source_result["security_clean_rate"],
        }
        selected, canonical = _selected_metrics(requested, values)
        result = {
            "schema_version": 2,
            "evaluation_scope": "project-held-out-fixtures",
            "split": split,
            "checkpoint": str(bundle.checkpoint_path),
            "tokenizer": str(bundle.tokenizer_path),
            "manifest": str(manifest_path),
            "corpus_artifact": str(corpus_path),
            "shard_manifest": str(shard_manifest_path) if shard_manifest_path else None,
            "metrics": selected,
            "completion": completion_result,
            "fim": fim_result,
            "generation": generation_cfg,
            **canonical,
        }
    if args.output:
        _write_json_atomic(_legacy._safe_project_path(root, args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_tokenizer(args) -> int:
    root = Path.cwd()
    config_path = _regular_path(args.config)
    cfg = _legacy._read_yaml(config_path, _legacy._TOKENIZER_KEYS)
    if cfg.get("type", "byte_level_bpe") != "byte_level_bpe":
        raise ValueError("only byte_level_bpe tokenizers are supported")
    configured_special = tuple(cfg.get("special_tokens", SPECIAL_TOKENS))
    if configured_special != tuple(SPECIAL_TOKENS):
        raise ValueError("configured special_tokens do not match the canonical token order")
    corpus_path = _safe_regular_project_path(
        root, cfg.get("corpus_artifact", "artifacts/corpus/dev/documents.jsonl")
    )
    manifest_path = _safe_regular_project_path(
        root, cfg.get("corpus_manifest", "manifests/dev.jsonl")
    )
    documents = _load_documents(corpus_path)
    training_documents = [item for item in documents if item["record"].get("split") == "train"]
    if not training_documents:
        raise ValueError("tokenizer corpus contains no training-split documents")
    texts = [item["text"] for item in training_documents]
    output = _legacy._safe_project_path(root, cfg.get("output_dir", "artifacts/tokenizers/dev"))
    output.mkdir(parents=True, exist_ok=True)
    metadata = train_tokenizer(
        texts,
        output / "tokenizer.json",
        vocab_size=int(cfg.get("vocab_size", 4096)),
        min_frequency=int(cfg.get("min_frequency", 2)),
    )
    trained_tokenizer = load_tokenizer(output / "tokenizer.json")
    invalid_special_ids = {
        token: trained_tokenizer.token_to_id(token)
        for expected_id, token in enumerate(SPECIAL_TOKENS)
        if trained_tokenizer.token_to_id(token) != expected_id
    }
    if invalid_special_ids:
        raise ValueError(
            "trained tokenizer special-token IDs do not match the canonical order: "
            f"{invalid_special_ids}"
        )
    metadata.update(
        {
            "seed": int(cfg.get("seed", 42)),
            "input_split": "train",
            "input_documents": len(texts),
            "input_bytes": sum(len(text.encode("utf-8")) for text in texts),
            "corpus_artifact": str(corpus_path),
            "corpus_artifact_sha256": sha256_file(corpus_path),
            "corpus_manifest": str(manifest_path),
            "corpus_manifest_sha256": sha256_file(manifest_path),
            "config_sha256": sha256_file(config_path),
        }
    )
    metadata["metrics"] = evaluate_tokenizer(trained_tokenizer, texts)
    _write_json_atomic(output / "metrics.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _validate_source_approvals(root: Path, manifest_path: Path) -> None:
    approval_path = _safe_regular_project_path(root, "manifests/approved-sources.jsonl")
    records = load_approval_manifest(approval_path)
    latest: dict[tuple[str, str], Any] = {}
    for record in records:
        latest[(record.source_uri, record.commit_sha)] = record
    identities: set[tuple[str, str]] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"derived manifest line {line_number} is not an object")
        if not value.get("included"):
            continue
        source_uri = value.get("source_uri")
        commit_sha = value.get("commit_sha")
        if not isinstance(source_uri, str) or not isinstance(commit_sha, str):
            raise ValueError(f"derived manifest line {line_number} lacks source identity")
        identities.add((source_uri, commit_sha))
    if not identities:
        raise ValueError("approval-required training manifest contains no included sources")
    missing = [identity for identity in sorted(identities) if identity not in latest]
    blocked = [
        identity
        for identity in sorted(identities)
        if identity in latest and latest[identity].status != "approved"
    ]
    if missing:
        raise ValueError(f"training sources are absent from the approval ledger: {missing}")
    if blocked:
        raise ValueError(f"training sources are not currently approved: {blocked}")


def _cmd_train(args) -> int:
    root = Path.cwd()
    config_path = _regular_path(args.config)
    cfg = _read_mapping(config_path)
    max_grad_norm = float(cfg.get("gradient_clip_norm", 1.0))
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    run_name = str(cfg.get("run_name", "dev"))
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single safe path component")
    output_root = _legacy._safe_project_path(root, cfg.get("output_root", "artifacts/runs"))
    run_dir = output_root / run_name
    model_path = _safe_regular_project_path(root, cfg.get("model_config", "configs/model/dev.yaml"))
    model_cfg = _read_mapping(model_path)
    if model_cfg.get("planning_only"):
        raise ValueError("planning-only model configurations cannot be trained")
    tokenizer_path = _safe_regular_project_path(
        root,
        Path(cfg.get("tokenizer_dir", "artifacts/tokenizers/dev")) / "tokenizer.json",
    )
    corpus_path = _safe_regular_project_path(
        root, cfg.get("corpus_artifact", "artifacts/corpus/dev/documents.jsonl")
    )
    manifest_path = _safe_regular_project_path(root, cfg.get("manifest", "manifests/dev.jsonl"))
    tokenizer = load_tokenizer(tokenizer_path)
    model_vocab = int(model_cfg.get("vocab_size", -1))
    tokenizer_vocab = int(tokenizer.get_vocab_size())
    if model_vocab != tokenizer_vocab:
        raise ValueError(
            f"tokenizer/model vocabulary mismatch: tokenizer={tokenizer_vocab}, model={model_vocab}"
        )
    invalid_special_ids = {
        token: tokenizer.token_to_id(token)
        for expected_id, token in enumerate(SPECIAL_TOKENS)
        if tokenizer.token_to_id(token) != expected_id
    }
    if invalid_special_ids:
        raise ValueError(
            "training tokenizer special-token IDs do not match the canonical order: "
            f"{invalid_special_ids}"
        )

    source_approval_hash: str | None = None
    authorization = None
    if cfg.get("approval_required"):
        source_approval_path = _safe_regular_project_path(root, "manifests/approved-sources.jsonl")
        source_approval_hash = sha256_file(source_approval_path)
        _validate_source_approvals(root, manifest_path)
        max_tokens = int(cfg.get("max_tokens", 0))
        if max_tokens > _EXPLICIT_TRAINING_AUTHORIZATION_THRESHOLD:
            authorization = require_training_authorization(
                root / "manifests" / "training-authorizations.jsonl",
                run_name=run_name,
                training_config=config_path,
                model_config=model_path,
                manifest=manifest_path,
                corpus=corpus_path,
                tokenizer=tokenizer_path,
                max_tokens=max_tokens,
            )

    lineage: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "training_config": str(config_path),
        "training_config_sha256": sha256_file(config_path),
        "model_config": str(model_path),
        "model_config_sha256": sha256_file(model_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_artifact": str(corpus_path),
        "corpus_artifact_sha256": sha256_file(corpus_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "source_approval_ledger_sha256": source_approval_hash,
        "training_authorization": asdict(authorization) if authorization else None,
        "bundle_verified": False,
    }
    _write_json_atomic(run_dir / "preflight-lineage.json", lineage)

    result = _legacy._cmd_train(args)
    if result == 0:
        checkpoint = run_dir / "checkpoints" / "latest"
        bundle = load_model_bundle(checkpoint)
        _verify_checkpoint_corpus(bundle, root, corpus_path)
        lineage.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "run_tokenizer": str(bundle.tokenizer_path),
                "run_tokenizer_sha256": sha256_file(bundle.tokenizer_path),
                "shard_manifest_identity": bundle.metadata.get("shard_manifest_hash"),
                "bundle_verified": True,
            }
        )
        _write_json_atomic(run_dir / "artifact-lineage.json", lineage)
    return result


def build_parser():
    return _legacy.build_parser()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "completion-evaluate":
        return _cmd_completion_evaluate(args)
    if args.command == "fim-evaluate":
        return _cmd_fim_evaluate(args)
    if args.command == "tokenizer" and args.tokenizer_command == "train":
        return _cmd_tokenizer(args)
    if args.command == "train":
        return _cmd_train(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
