import pytest

from ts_coder.sandbox import SandboxUnavailable, run_in_sandbox


def test_external_execution_is_refused():
    with pytest.raises(SandboxUnavailable):
        run_in_sandbox(["node", "unknown.js"])
