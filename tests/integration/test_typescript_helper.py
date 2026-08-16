import shutil
import subprocess
from pathlib import Path

import pytest

from ts_coder.evaluation.compile import compile_typescript


@pytest.mark.skipif(shutil.which("npm") is None, reason="node unavailable")
def test_typescript_helper_builds():
    root = Path(__file__).parents[2] / "tools" / "typescript"
    if not (root / "package.json").exists():
        pytest.skip("helper lane not yet present")
    result = subprocess.run(
        ["npm", "run", "build", "--if-present"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_helper_treats_filename_as_display_metadata():
    root = Path(__file__).parents[2] / "tools" / "typescript"
    helper = root / "dist" / "parse.js"
    if not helper.exists():
        pytest.skip("helper not built")
    result = subprocess.run(
        ["node", str(helper)],
        input='{"filename":"../../outside.ts","source":"const x = 1;"}',
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert '"success":true' in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_helper_compiles_controlled_tsx_without_react_runtime():
    root = Path(__file__).parents[2] / "tools" / "typescript"
    if not (root / "dist" / "compile.js").exists():
        pytest.skip("helper not built")
    result = compile_typescript(
        'export function Badge() { return <span className="ok">ok</span>; }',
        filename="Badge.tsx",
    )
    assert result["success"] is True, result
