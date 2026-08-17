# Security Policy

Report vulnerabilities privately to the maintainers. Do not publish credentials,
exploit details, private corpus content, or unreviewed model artifacts.

Repositories, derived corpora, tokenizers, shards, and model artifacts are untrusted.
Never execute harvested code, install its dependencies, deserialize unknown pickle
payloads, expose credentials, mount the Docker socket, or grant evaluation workloads
unrestricted network access. Controlled project fixtures may be parsed and compiled
locally. External code execution requires the sandbox described in
`docs/THREAT_MODEL.md`.

Probable secrets cause source rejection. Logs contain detector category and a
redacted fingerprint, never the matched value. Streaming loaders verify shard record
counts and SHA-256 descriptors before training. Checkpoints use tensor-only loading
and must match the recorded tokenizer, vocabulary, model configuration, manifest,
and shard identity.

Source approval does not authorize an expensive training run or weight release.
Approval-required runs verify current source decisions, and runs above the bounded
pilot threshold require an exact append-only training authorization. See
`docs/TRAINING_AUTHORIZATION.md` and `docs/DATA_POLICY.md`.
