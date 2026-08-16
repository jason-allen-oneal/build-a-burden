from ts_coder.evaluation.fim import exact_match, token_accuracy
from ts_coder.evaluation.memorization import longest_common_substring
from ts_coder.evaluation.security import contains_secret


def test_metrics_and_redaction_boundary():
    assert exact_match([1], [1])
    assert token_accuracy([1, 2], [1, 3]) == 0.5
    assert longest_common_substring("abc", "zab") == 2
    assert contains_secret("token=EXAMPLE_NOT_A_SECRET") is False
    assert contains_secret("-----BEGIN PRIVATE KEY-----")


def test_unified_evaluation_reports_secret_findings():
    from ts_coder.evaluation.runner import evaluate_sources

    result = evaluate_sources(['const apiKey = "EXAMPLE_NOT_A_SECRET_WITH_SUFFIX";'])
    assert result["security_clean_rate"] == 0.0
    assert result["security_findings"] == [{"index": 0, "category": "secret-pattern"}]
