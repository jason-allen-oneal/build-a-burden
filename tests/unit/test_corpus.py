from pathlib import Path
from ts_coder.corpus.deduplicate import near_duplicate, normalize_typescript
from ts_coder.corpus.ingest import ingest_repository
from ts_coder.corpus.repository_pack import pack_repository
from ts_coder.corpus.split import assign_splits

FIXTURE = Path(__file__).parents[2] / "fixtures/repositories/basic"


def test_ingest_is_deterministic_and_filters() -> None:
    kwargs = {"retrieved_at": "2026-01-01T00:00:00Z", "pipeline_version": "test"}
    first, content = ingest_repository(FIXTURE, **kwargs)
    second, _ = ingest_repository(FIXTURE, **kwargs)
    assert first == second
    by_path = {r["relative_path"]: r for r in first}
    assert by_path["src/service.ts"]["included"]
    assert "generated-header" in by_path["src/client.generated.ts"]["exclusion_reasons"]
    assert by_path["src/credential.ts"]["secret_scan_status"] == "rejected"
    assert not by_path["README.md"]["included"]
    assert by_path["package.json"]["included"]
    assert by_path["package.json"]["source_type"] == "metadata"
    assert (
        sum(
            r["included"]
            for r in first
            if r["normalized_sha256"] == by_path["src/types.ts"]["normalized_sha256"]
        )
        == 1
    )


def test_near_duplicate_and_pack_safety() -> None:
    assert normalize_typescript("//x\nconst x = 1") == "const x = 1"
    assert near_duplicate("const a=1; const b=2;", "// c\nconst a=1; const b=2;")
    assert pack_repository([("src/a.ts", "export const a=1;")]).startswith("<repo>")
    try:
        pack_repository([("../escape.ts", "x")])
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal accepted")


def test_approved_scope_is_enforced_but_all_files_are_recorded() -> None:
    records, _ = ingest_repository(
        FIXTURE,
        retrieved_at="2026-01-01T00:00:00Z",
        pipeline_version="test",
        approved_scope=("include:src/service.ts",),
    )
    by_path = {record["relative_path"]: record for record in records}
    assert by_path["src/service.ts"]["included"]
    assert not by_path["src/types.ts"]["included"]
    assert "outside-approved-scope" in by_path["src/types.ts"]["exclusion_reasons"]
    assert "LICENSE" in by_path


def test_compiler_harness_is_excluded_but_ordinary_tests_are_kept(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy",
        encoding="utf-8",
    )
    harness = tmp_path / "tests" / "cases" / "fourslash" / "completion.ts"
    harness.parent.mkdir(parents=True)
    harness.write_text(
        "// @Filename: /index.ts\n"
        "//// export const value = 1;\n"
        'verify.completions({ marker: "" });\n',
        encoding="utf-8",
    )
    ordinary = tmp_path / "tests" / "service.test.ts"
    ordinary.write_text(
        'import { describe, it } from "vitest";\n'
        'describe("service", () => { it("works", () => {}); });\n',
        encoding="utf-8",
    )
    records, _ = ingest_repository(tmp_path)
    by_path = {record["relative_path"]: record for record in records}
    assert not by_path["tests/cases/fourslash/completion.ts"]["included"]
    assert (
        "test-harness-filename-directive"
        in by_path["tests/cases/fourslash/completion.ts"]["exclusion_reasons"]
    )
    assert by_path["tests/service.test.ts"]["included"]


def test_real_source_splits_keep_repository_families_together() -> None:
    records = [
        {"repository_id": "repo-a", "dedup_cluster": "shared", "included": True},
        {"repository_id": "repo-a", "dedup_cluster": "a-only", "included": True},
        {"repository_id": "repo-b", "dedup_cluster": "b-only", "included": True},
        {"repository_id": "repo-c", "dedup_cluster": "c-only", "included": True},
        {"repository_id": "repo-d", "dedup_cluster": "shared", "included": True},
    ]
    assign_splits(records, group_by_repository=True)
    by_repository = {
        repository: {record["split"] for record in records if record["repository_id"] == repository}
        for repository in {record["repository_id"] for record in records}
    }
    assert all(len(splits) == 1 for splits in by_repository.values())
    assert by_repository["repo-a"] == by_repository["repo-d"]
    assert by_repository["repo-a"] == {"train"}
    assert {next(iter(splits)) for splits in by_repository.values()} == {
        "train",
        "validation",
        "test",
    }
