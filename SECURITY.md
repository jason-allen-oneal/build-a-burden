# Security Policy

Report vulnerabilities privately to the maintainers. Do not publish credentials,
exploit details, or private corpus content.

Repositories and model artifacts are untrusted. Never execute harvested code,
install its dependencies, deserialize unknown pickle payloads, expose credentials,
mount the Docker socket, or grant evaluation workloads unrestricted network access.
Controlled Milestone 1 fixtures may be parsed and compiled locally; external code
requires the sandbox described in `docs/THREAT_MODEL.md`.

Probable secrets cause rejection. Logs contain detector category and a redacted
fingerprint, never the match. Load checkpoints only from trusted runs and prefer
safe tensor-only release formats. See `docs/DATA_POLICY.md` for corpus controls.
