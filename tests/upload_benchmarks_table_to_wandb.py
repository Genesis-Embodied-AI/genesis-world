"""
Upload benchmark results to Weights & Biases.

This script parses benchmark results files (memory or performance) generated
by monitor_test_mem.py or similar monitoring tools and uploads them to W&B.

Memory example:
env=franka 	| constraint_solver=None 	| gjk_collision=True 	| batch_size=30000 	| backend=cuda 	| dtype=field 	| max_mem_mb=123
env=anymal 	| constraint_solver=Newton 	| gjk_collision=False 	| batch_size=0 	| backend=cpu 	| dtype=ndarray 	| max_mem_mb=234

Performance example:
env=franka 	| batch_size=30000 	| dtype=field 	| backend=cuda 	| compile_time=68.4 	| runtime_fps=20067534.0 	| realtime_factor=200675.3
env=anymal 	| constraint_solver=Newton 	| gjk_collision=False 	| batch_size=0 	| backend=cpu 	| dtype=ndarray 	|compile_time=3.2 	| runtime_fps=1355.0 	| realtime_factor=3322

... and check uploads to https://wandb.ai/genesis-ai-company/genesis-benchmarks-mem/table
"""

import argparse
import wandb
import os
import sys
from pathlib import Path

from utils.assets import get_git_commit_info
from utils.misc import pprint_oneline


def upload_results_to_wandb(
    run_prefix: str | None, results_file_path: str, project_name: str, metric_names=None
) -> None:
    """
    Parse results file in pipe-delimited format and upload to W&B.

    Args:
        results_file_path: Path to the results file
        project_name: W&B project name (e.g., "genesis-benchmarks-mem" or "genesis-benchmarks-perf")
        metric_names: List of metric field names to log. If None, logs all non-parameter fields.
    """
    revision, _ = get_git_commit_info()
    print(f"Uploading results to W&B project '{project_name}' for revision: {revision}")

    uploaded_count = 0
    skipped_count = 0

    # Initialize a single run for all benchmark results
    name = f"{revision[:12]}"
    if run_prefix:
        name = f"{run_prefix}-{name}"
    run = wandb.init(
        project=project_name,
        name=name,
        config={
            "revision": revision,
        },
        settings=wandb.Settings(
            x_disable_stats=True,
            console="off",
        ),
    )

    with open(results_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse pipe-delimited format: key=value | key=value | ...
            params = {}
            for part in line.split(" \t| "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k.strip()] = v.strip()

            if not params:
                skipped_count += 1
                continue

            # Extract metrics based on specified metric_names or all remaining fields
            if metric_names:
                metrics = {k: float(params.pop(k)) for k in metric_names if k in params}
            else:
                # Extract all numeric fields as metrics (non-parameters)
                metrics = {}
                for k in list(params.keys()):
                    try:
                        metrics[k] = float(params[k])
                    except ValueError:
                        continue
                    del params[k]

            if not metrics:
                skipped_count += 1
                continue

            # Sort params for consistent benchmark ID ordering
            sorted_params = dict(sorted(params.items()))

            # Create benchmark ID matching alarm.yml format
            benchmark_id_suffix = pprint_oneline(sorted_params, delimiter="-")

            for metric_name, metric_value in metrics.items():
                benchmark_id = f"{metric_name}-{benchmark_id_suffix}"
                print(f"📊 Uploading {benchmark_id}: {metric_value}")
                run.log({benchmark_id: metric_value})

            uploaded_count += 1

    run.finish()

    print(f"\n✅ Upload complete: {uploaded_count} results processed, {skipped_count} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload benchmark results to W&B")
    parser.add_argument("--in-file", required=True, help="Path to results file")
    parser.add_argument(
        "--project", required=True, help="W&B project name (e.g., genesis-benchmarks-mem or genesis-benchmarks-perf)"
    )
    parser.add_argument("--run-prefix", help="Added at start of W&B run name, if provided")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric field names to upload (e.g., max_mem_mb compile_time runtime_fps). If not specified, all numeric fields are uploaded.",
    )
    args = parser.parse_args()
    upload_results_to_wandb(
        run_prefix=args.run_prefix, results_file_path=args.in_file, project_name=args.project, metric_names=args.metrics
    )
