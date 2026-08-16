from pathlib import Path
from ts_coder.corpus.secret_filter import scan_secrets


def test_secret_results_are_redacted() -> None:
    value = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
    status, findings = scan_secrets(Path("source.ts"), f'const token = "{value}";')
    assert status == "rejected" and findings
    assert value not in repr(findings)


def test_explicit_nonsecret_marker_is_allowed() -> None:
    status, _ = scan_secrets(Path("fixture.ts"), 'const token = "TEST_TOKEN_DO_NOT_USE";')
    assert status == "clean"


def test_marker_prefix_does_not_exempt_suffix() -> None:
    status, _ = scan_secrets(
        Path("fixture.ts"), 'const token = "TEST_TOKEN_DO_NOT_USE_WITH_SUFFIX";'
    )
    assert status == "rejected"


def test_marker_concatenation_does_not_exempt_following_value() -> None:
    status, _ = scan_secrets(
        Path("fixture.ts"), 'const token = "TEST_TOKEN_DO_NOT_USE" + "secret-suffix";'
    )
    assert status == "rejected"


def test_evaluation_secret_marker_prefix_does_not_exempt_suffix() -> None:
    from ts_coder.evaluation.security import contains_secret

    assert contains_secret("apiKey = 'EXAMPLE_NOT_A_SECRET_WITH_SUFFIX'")
