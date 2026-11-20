"""Check explicit compiler baselines independently of the simulation test suite."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "alarm.py"


class ExplicitBaselineTests(unittest.TestCase):
    def run_comparison(self, root, baseline_fps="100", candidate_fps="80", baseline_env="anymal"):
        for name, fps, env in (("baseline", baseline_fps, baseline_env), ("candidate", candidate_fps, "anymal")):
            directory = root / name
            directory.mkdir()
            (directory / "speed_test.txt").write_text(
                f"env={env} | dtype=ndarray | runtime_fps={fps} | compile_time=10 | realtime_factor=1\n"
            )
            (directory / "mem_test.txt").write_text(f"env={env} | dtype=ndarray | max_mem_mb=100\n")
        command = [
            sys.executable,
            str(SCRIPT),
            "--baseline-artifacts-dir",
            str(root / "baseline"),
            "--baseline-revision",
            "quadrants-main",
            "--speed-artifacts-dir",
            str(root / "candidate"),
            "--mem-artifacts-dir",
            str(root / "candidate"),
            "--check-body-path",
            str(root / "report.md"),
            "--runtime-fps-regression-tolerance-pct",
            "8",
        ]
        for metric in ("runtime-fps", "compile-time", "mem"):
            command.extend([f"--csv-{metric}-path", str(root / f"{metric}.csv")])
        environment = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
        return subprocess.run(command, env=environment, text=True, capture_output=True)

    def test_single_baseline_detects_regression_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_comparison(root)
            self.assertEqual(result.returncode, 42, result.stderr)
            report = (root / "report.md").read_text()
            self.assertIn("quadrants-main", report)
            self.assertNotIn("NaN", report)

    def test_matching_measurements_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_comparison(Path(directory), candidate_fps="100")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_configuration_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_comparison(Path(directory), baseline_env="other")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Baseline lacks", result.stderr)

    def test_nonfinite_baseline_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_comparison(Path(directory), baseline_fps="nan")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid measurement", result.stderr)


if __name__ == "__main__":
    unittest.main()
