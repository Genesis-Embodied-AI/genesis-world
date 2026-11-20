"""Check explicit compiler baselines independently of the simulation test suite."""

# Currently run manually, not as part of CI:
# python -m pytest -o addopts= .github/workflows/scripts/tests/test_alarm_baseline.py -v

import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import wandb
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "alarm.py"
WORKFLOW = SCRIPT.parents[1] / "alarm.yml"


@pytest.fixture
def measurements(tmp_path):
    for name in ("baseline-artifacts", "artifacts"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "speed_test.txt").write_text(
            "env=anymal | dtype=ndarray | runtime_fps=100 | compile_time=10 | realtime_factor=1\n"
        )
        (directory / "mem_test.txt").write_text("env=anymal | dtype=ndarray | max_mem_mb=100\n")
    return tmp_path


@pytest.fixture
def command(measurements):
    arguments = [
        sys.executable,
        str(SCRIPT),
        "--baseline-artifacts-dir",
        str(measurements / "baseline-artifacts"),
        "--baseline-revision",
        "quadrants-main",
        "--speed-artifacts-dir",
        str(measurements / "artifacts"),
        "--mem-artifacts-dir",
        str(measurements / "artifacts"),
        "--check-body-path",
        str(measurements / "report.md"),
        "--runtime-fps-regression-tolerance-pct",
        "8",
    ]
    for metric in ("runtime-fps", "compile-time", "mem"):
        arguments.extend([f"--csv-{metric}-path", str(measurements / f"{metric}.csv")])
    return arguments


@pytest.mark.parametrize("fps, exit_code", [(100, 0), (80, 42), (120, 43)])
def test_explicit_baseline_without_credentials(command, measurements, fps, exit_code):
    candidate = measurements / "artifacts" / "speed_test.txt"
    candidate.write_text(candidate.read_text().replace("runtime_fps=100", f"runtime_fps={fps}"))
    environment = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert result.returncode == exit_code, result.stderr
    report = (measurements / "report.md").read_text()
    assert "quadrants-main" in report
    assert "NaN" not in report


@pytest.mark.parametrize(
    "replacement, message",
    [("env=other", "Baseline lacks"), ("runtime_fps=nan", "Invalid measurement")],
)
def test_invalid_baseline_cannot_pass(command, measurements, replacement, message):
    baseline = measurements / "baseline-artifacts" / "speed_test.txt"
    original = "env=anymal" if replacement.startswith("env=") else "runtime_fps=100"
    baseline.write_text(baseline.read_text().replace(original, replacement))
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode not in (0, 42, 43)
    assert message in result.stderr


def test_default_comparison_still_reads_history(command, monkeypatch):
    del command[2:6]
    monkeypatch.setattr(sys, "argv", command[1:])
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_ENTITY", "test-only")

    def read_history():
        raise RuntimeError("historical baseline requested")

    monkeypatch.setattr(wandb, "Api", read_history)
    with pytest.raises(RuntimeError, match="historical baseline requested"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize("revision, expected_code", [("quadrants-main", 0), ("", 1)])
def test_workflow_analysis_passes_explicit_baseline(measurements, revision, expected_code):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(step for step in workflow["jobs"]["comment-if-regressed"]["steps"] if step.get("id") == "analyze")
    candidate = measurements / "artifacts" / "speed_test.txt"
    candidate.write_text(candidate.read_text().replace("runtime_fps=100", "runtime_fps=80"))
    shell = step["run"].replace("/tmp", str(measurements / "outputs"))
    shell = shell.replace(
        "python .github/workflows/scripts/alarm.py", f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT))}"
    )
    environment = dict(
        os.environ,
        BASELINE_ARTIFACT="baseline",
        BASELINE_REVISION=revision,
        GITHUB_ENV=str(measurements / "github-env"),
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", shell], cwd=measurements, env=environment, text=True, capture_output=True
    )
    assert result.returncode == expected_code, result.stderr
    if revision:
        output = (measurements / "github-env").read_text()
        assert "HAS_REGRESSIONS=1" in output
        assert "HAS_ERRORS=0" in output
        assert "quadrants-main" in output
    else:
        assert "baseline_revision is required" in result.stdout
