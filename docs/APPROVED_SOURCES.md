# Approved-source intake

This ledger is the gate between local intake and any approved corpus. Approval is a
separate human decision. The acquisition command acts only on exact approved
URI/revision pairs and never executes repository code.

## Required decision

Each source needs an append-only record in
`manifests/approved-sources.jsonl` containing:

- stable `source_id`;
- credential-free `source_uri`;
- exact 40-character lowercase `commit_sha`;
- detected SPDX license from the conservative allowlist;
- `approved_by`, UTC `approved_at`, and a path/scope list;
- status `approved`, `review`, `rejected`, or `removed`.

The only exception is the project-authored `fixture://` source with revision
`fixture`. Real public or private repositories must use their exact immutable
revision. Query strings, fragments, signed URLs, and embedded credentials are
rejected rather than persisted.

## Acquisition and dry-run procedure

1. Add a reviewed approval record with machine-readable `include:<glob>` and
   `exclude:<glob>` scope entries. Update the version-controlled
   `approval_manifest_sha256` in the approved data config.
2. Add the immutable snapshot and receipt paths to the data config.
3. Validate the acquisition plan without network access or writes:

   ```bash
   uv run python -m ts_coder.cli corpus approvals validate \
     --manifest manifests/approved-sources.jsonl
   uv run python -m ts_coder.cli corpus acquire \
     --config configs/data/approved.yaml --dry-run
   ```

4. Acquire one approved source at a time, then run its independent corpus dry-run:

   ```bash
   uv run python -m ts_coder.cli corpus acquire \
     --config configs/data/approved.yaml --source-id SOURCE_ID
   uv run python -m ts_coder.cli corpus build \
     --config configs/data/approved.yaml --source-id SOURCE_ID --dry-run
   ```

5. Review license, rejection, duplicate, secret, and split statistics. Run the
   combined dry-run before writing the derived manifest and documents.

Acquisition uses an isolated bare Git repository, verifies the exact fetched commit,
enumerates and validates Git tree entries, and materializes bounded blobs without a
checkout. Submodules and special modes are rejected. Internal relative symlinks are
permitted only when their targets remain inside the snapshot; they are always
rejected from the corpus and fingerprinted in the receipt. Snapshot mutation causes
receipt verification to fail closed.

The build requires an exact approved URI/revision match when
`require_approval: true`. A later removal is appended as a `removed` decision;
existing records are never edited in place. Rebuild affected manifests and
shards after removal, and retain the prior hashes for audit.

## Current state — 2026-08-15

Seven exact revisions are approved and acquired: `colinhacks-zod`, `vitejs-vite`,
`nestjs-nest`, `typescript-eslint-typescript-eslint`, `typeorm-typeorm`,
`reduxjs-redux-toolkit`, and `microsoft-typescript`. The post-filter combined
manifest contains 99,427 audited records and 15,070 included documents, with
7,376,774 tokenizer tokens. Compiler/fourslash harness records are rejected from
the derived corpus while remaining auditable in the source manifest. Generated
snapshots, receipts, manifests, and tokenizers remain ignored artifacts. Approval
does not authorize redistribution, model publication, or larger training.
