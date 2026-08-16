# Threat Model

| Threat / asset | Attacker and entry | Impact | Mitigation | Remaining risk / validation |
|---|---|---|---|---|
| Malicious repositories / host | source author via files or scripts | execution, theft | acquire exact commits through isolated bare Git; no checkout, hooks, filters, submodules, installs, or source execution; sandbox external evaluation | parser/Git bugs; adversarial fixtures and integration tests |
| Git transport exhaustion / disk | approved hostile commit via fetch pack | disk, network, CPU, or memory exhaustion | shallow blob-filtered fetch, disabled lazy fetch, per-file OS quota, 180-second timeout, bounded output, disposable staging | quota is per file rather than aggregate cgroup/network accounting; canonical GitHub intake only; resource-boundary tests |
| Poisoning / model behavior | corpus contributor via source | backdoors, bad code | provenance, quality review, family splits, removal rebuild | subtle poison; distribution and behavior review |
| Credentials / secrets | accidental committer via content | disclosure, memorization | pattern/entropy scan, reject file, redacted logs | novel formats; seeded fake-secret tests |
| Provenance falsification | source or pipeline operator via metadata | unlawful/untraceable data | exact revision, strict snapshot receipt, tree/license hashes, append-only decisions, config-pinned ledger hash | upstream lies; ledger hash is not a signature if config and ledger are both compromised; sampled manual audit |
| Benchmark contamination | corpus overlap | misleading scores | quarantine, hash and near-duplicate checks | semantic clones; contamination report |
| Package supply chain / CI | dependency publisher | code execution | pinned direct deps and lockfiles, audits, `npm ci --ignore-scripts` | transitive compromise; audit and lock review |
| Container escape / host | malicious evaluated code | host takeover | no network/privilege/socket/secrets, non-root, limits, read-only root | runtime zero-day; sandbox tests and patching |
| Checkpoint tampering / process | artifact provider | corrupt result, resource exhaustion | tensor-only `weights_only` loading, strict envelope/schema checks, file/tensor-size bounds, recorded tokenizer/manifest hashes | no signature or remote provenance verification; only load trusted run artifacts |
| Traversal, symlink, bombs / disk | crafted archive | overwrite or exhaustion | reject absolute/`..`/escaping links; byte/file/depth limits; disposable extraction | parser edge cases; malicious archive tests |
| Oversized input / availability | source author | memory/disk exhaustion | streaming, file and corpus limits, timeouts | compressed edge cases; resource tests |
| Log injection / audit trail | crafted source text | concealed events, terminal control | structured logs, control escaping, no full source by default | downstream renderer bugs; log tests |
| Memorization / corpus privacy | training data contributor | source reproduction | conservative corpus, removal, matching metrics, release review | non-exact recall; red-team generations |
| Vulnerable generated code / users | model output | downstream compromise | compilation/tests/security evaluation and human review | incomplete detectors; security task suite |
| CI secret exposure / credentials | malicious test or dependency | account compromise | CI has no secrets; never execute unknown repositories | platform compromise; workflow review |

External compilation/test containers must disable networking, run non-root, use a
read-only root where practical, provide only a temporary writable directory, enforce
CPU/memory/process/file/time limits, avoid sensitive mounts and the Docker socket, pin
the image, and restrict syscalls. Milestone 1 local compilation is limited to
project-authored fixtures.
