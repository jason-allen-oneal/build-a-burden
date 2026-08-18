#!/usr/bin/env python3
"""Append an exact structured decision for a costly training run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from ts_coder.training.authorization import (
    append_training_authorization,
    make_training_authorization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="manifests/training-authorizations.jsonl")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--status", choices=("approved", "rejected", "revoked"), default="approved")
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    record = make_training_authorization(
        run_name=args.run_name,
        training_config=args.training_config,
        model_config=args.model_config,
        manifest=args.manifest,
        corpus=args.corpus,
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
        status=args.status,
        notes=args.notes,
    )
    append_training_authorization(args.ledger, record)
    print(json.dumps(asdict(record), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
