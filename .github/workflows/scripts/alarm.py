"""
This script runs from alarm.yml

Terminology/variable names:
- benchmark suite results: the results of running all benchmark tests once, for a specific code base
- the code base could be conceptually:
    - the current code under test
    - some past revision of the code, described by a git commit hash
- there are actually multiple benchmark test suites, identified by a suite_id
    - in this script, we are only interested in the rigid benchmark suite
- metric: the string name of something we are measuring, such as 'runtime_fps'
- configuration parameter: something we vary/control, such as batch_size, or env
- config_params_str: a string like "backend=cpu-n_envs=64", which specifies specific configuration
  parameters, in string format
      - note that, two config_params_str might represent the same configuration, but be different strings,
        because ordering of configuration parameters might be differnet
- config_params_fdict: a frozen dict that represents a specific set of configuration parameters
      - by comparison with config_str, two identical config_params_fdict's always represent the same configuration
      - note that config_params_fdict's are hashable
      - (fdict is an abbreviation for 'frozendict')
- config_param_names: ordered list of the config parameter names, that we have almost certainly derived
  from a config_params_fdict, by simply returning the ordered list of keys (though we may have merged
  such a list over multiple config_params_fdict's)
  - we are prefixing with 'config' to make explicit that this does not include the names of metrics
- 'pipeline format':
   a string having format like:
   "solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=990.0 | realtime_factor=49.5"
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from frozendict import frozendict
from wandb.apis.public import Run

import wandb


def config_params_str_to_fdict(config_params_str: str) -> frozendict[str, str]:
    """
    Expects a config_params_str in the string format like:
    solver=PBD-backend=cpu-n_envs=128

    Returns this as a frozen dict of key value pairs.

    Note that the values are strings, not converted into numbers.
    """
    kv = {}
    if config_params_str:
        for token in config_params_str.split("-"):
            token = token.strip()
            if token and "=" in token:
                k, v = token.split("=", 1)
                kv[k.strip()] = v.strip()
    return frozendict(kv)


def merge_string_tuples(tuples: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """
    Merge tuples of strings into a single tuple of strings which:
    - preserves the relative order of keys within each tuple
    - gives precedence to later tuples when conflicts arise
    """
    merged_keys = list(tuples[-1])
    merged_keys_set = set(merged_keys)
    for tuple_ in tuples[:-1]:
        for key in tuple_:
            if key not in merged_keys_set:
                merged_keys.append(key)
                merged_keys_set.add(key)
    return tuple(merged_keys)


class SortKey:
    def __init__(self, config_param_names: Iterable[str]) -> None:
        self.config_param_names = config_param_names

    def __call__(self, d: frozendict[str, Any]) -> list[tuple[int, int | float | None]]:
        """
        returns list of tuples that can be used to order
        dictionaries of values. The sort key function returns
        a list of tuples of (True|False, value | None), where the sequence of
        (True|False, value) matches that of config_param_names and:
        - only keys in config_param_names are considered in the sorting
        (in the context of this script, this lets us ignore the values of
        metrics during sorting)
        - when a param_name is present in the dictionary, the tuple
        contains (False, value), otherwise (True, None)

        Since the resulting tuples will be used for sorting, the result
        is that we will first sort the incoming dictionaries by the first param_name,
        then the second, etc
        - for a particular param_name, the dicts without that param_name will
        be placed after the dicts with that param name, since True is after False.
        """
        key_list = []
        for col in self.config_param_names:
            val = d.get(col)
            key_list.append((val is None, val))
        return key_list


def parse_results_file(
    results_file_path: Path, metric_keys: Iterable[str]
) -> dict[frozendict[str, str], dict[str, float]]:
    """
    results file path should have lines in pipeline format, like:
    solver=PBD | backend=cpu | n_envs=128 | compile_time=2.52 | runtime_fps=990.0 | realtime_factor=49.5
    solver=PBD | backend=gpu | n_envs=1024 | compile_time=2.54 | runtime_fps=985.0 | realtime_factor=49.3
    solver=MPM | backend=cpu | n_envs=64 | compile_time=2.53 | runtime_fps=988.0 | realtime_factor=49.4

    This function returns a dict of dicts, something like:
    {
        FrozenDict({"solver": "PBD", "backend": "cpu"}): {
            "compile_time": 2.52,
            "runtime_fps": 990.0,
        }
    }
    So:
    - the keys of the top level dict are frozendict's representing all the key value pairs in a results row
      EXCEPT the metric key value pairs
    - the values are dicts where the keys are names of the metrics in metric_keys, and the values are
      the measured value of that metric

    Conceptually the keys are config_param_fdict's, and the values are a dictionary of metric names and
    values.
    """
    # easy to accidentally send a string instead of a tuple
    assert isinstance(metric_keys, tuple)
    results: dict[frozendict[str, str], dict[str, int | float]] = {}
    for line in results_file_path.read_text().splitlines():
        config_param_dict: dict[str, str] = dict(  # type: ignore
            map(str.strip, p.split("=", 1))
            for p in line.split("|")
            if "=" in p  # type: ignore
        )
        metrics: dict[str, float | int] = {}
        for k in metric_keys:
            try:
                # removes metric keys from the config param dict, and adds to the metric kv dict
                metrics[k] = float(config_param_dict.pop(k))
            except (ValueError, TypeError, KeyError):
                pass
        config_param_fdict: frozendict[str, str] = frozendict(config_param_dict)
        results[config_param_fdict] = metrics
    return results


def fmt_num(v, is_int: bool):
    """
    converts number to string where:
    - ints => displays as int
    - floats => displays to 2 decimal places
    """
    if v != v:
        return "NaN"
    return f"{int(v):,}" if is_int else f"{v:.2f}"


class WandbParser:
    @property
    def project(self):
        raise NotImplementedError()

    def __call__(
        self,
        benchmark_under_test: "BenchmarkRunUnderTest",
        records_by_commit_hash: dict[str, dict[frozendict[str, str], dict[str, int | float]]],
        config,
        summary,
        commit_hash: str,
    ) -> None:
        raise NotImplementedError()


class WandbParserNewFormat(WandbParser):
    @property
    def project(self):
        return "genesis-benchmarks-2"

    def __call__(
        self,
        benchmark_under_test: "BenchmarkRunUnderTest",
        records_by_commit_hash: dict[str, dict[frozendict[str, str], dict[str, int | float]]],
        config,
        summary,
        commit_hash: str,
    ) -> None:
        for k, v in summary.items():
            if k.startswith("_"):
                continue
            metric_name, _, kv_pairs_str = k.partition("-")
            kv_pairs_fdict = config_params_str_to_fdict(kv_pairs_str)
            records_by_commit_hash[commit_hash][kv_pairs_fdict][metric_name] = v


class BenchmarkRunUnderTest:
    """
    This class contains the data about the benchmark run under test, which we will then
    compare with historical data. This data is loaded from text files in pipe format.
    | foo=123 | bar=456 | ...
    """

    def __init__(self, artifacts_dir: Path, metric_keys: Iterable[str], filename_glob: str) -> None:
        """
        metric_keys: the keys corresponding to values being measured, such as runtime_fps
        filename_glob: how to locate the data files with the data for the benchmark run
        under test.
        """
        self.result_file_paths = list(artifacts_dir.rglob(filename_glob))
        # make sure we do actually have some current benchmark data to read
        assert self.result_file_paths

        self.metric_keys = metric_keys

        # self.results is a dictionary where the keys are config_param_fdict's, and the values
        # are dicts of metric names and values
        self.results: dict[frozendict[str, str], dict[str, float]] = {}
        for self.result_file_path in self.result_file_paths:
            self.results |= parse_results_file(self.result_file_path, self.metric_keys)
        # all the config_param_fdicts that we need to check for a 'complete set', when looking
        # at historical data (some earlier runs might be missing some of the newer benchmark
        # runs)
        self.all_config_param_fdicts = frozenset(self.results.keys())
        assert self.all_config_param_fdicts

        # ordered list of the config parameter names
        self.config_param_names = merge_string_tuples(tuple((tuple(kv.keys())) for kv in self.results.keys()))


class Alarm:
    def __init__(self, args: argparse.Namespace) -> None:
        self.max_valid_revisions = args.max_valid_revisions
        self.max_fetch_revisions = args.max_fetch_revisions

        # let's just define these in one place
        self.metric_compile_time = "compile_time"
        self.metric_runtime_fps = "runtime_fps"
        self.metric_realtime_factor = "realtime_factor"
        self.metric_max_mem_mb = "max_mem_mb"

        self.metrics_tol = {
            self.metric_runtime_fps: args.runtime_fps_regression_tolerance_pct,
            self.metric_compile_time: args.compile_time_regression_tolerance_pct,
            self.metric_max_mem_mb: args.mem_regression_tolerance_pct,
        }

        self.speed_artifacts_dir = Path(args.speed_artifacts_dir).expanduser().resolve()
        self.mem_artifacts_dir = Path(args.mem_artifacts_dir).expanduser().resolve()
        self.check_body_path = Path(args.check_body_path).expanduser()

        self.csv_out_file_by_metric_name = {
            self.metric_runtime_fps: Path(args.csv_runtime_fps_path).expanduser().resolve(),
            self.metric_compile_time: Path(args.csv_compile_time_path).expanduser().resolve(),
            self.metric_max_mem_mb: Path(args.csv_mem_path).expanduser().resolve(),
        }

        self.speed_metric_keys = (
            self.metric_compile_time,
            self.metric_runtime_fps,
            self.metric_realtime_factor,
        )
        self.mem_metric_keys = (self.metric_max_mem_mb,)  # note: make sure is a tuple

        self.dev_skip_speed = args.dev_skip_speed
        self.dev_allow_all_branches = args.dev_allow_all_branches

        assert "WANDB_API_KEY" in os.environ

        self.wandb_entity = os.environ["WANDB_ENTITY"]

    def fetch_wandb_data(
        self,
        benchmark_under_test: BenchmarkRunUnderTest,
        run_name_prefix: str | None,
        wandb_parser: WandbParser,
    ) -> dict[str, dict[frozendict[str, str], dict[str, float | int]]]:
        api = wandb.Api()
        runs_iter: Iterable[Run] = api.runs(f"{self.wandb_entity}/{wandb_parser.project}", order="-created_at")

        commit_hashes = set()
        records_by_commit_hash: dict[str, dict[frozendict[str, str], dict[str, float | int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for i, run in enumerate(runs_iter):
            if run_name_prefix and not run.name.startswith(run_name_prefix):
                continue

            # Abort if still not complete after checking enough runs.
            # This would happen if a new benchmark has been added, and not enough past data is available yet.
            if len(commit_hashes) == self.max_fetch_revisions:
                break

            # Early return if enough complete records have been collected
            complete_records = [
                benchmark_under_test.all_config_param_fdicts.issubset(record.keys())
                for record in records_by_commit_hash.values()
            ]
            if sum(complete_records) == self.max_valid_revisions:
                break

            # Load config and summary, with support of legacy runs
            summary: dict[str, Any]
            try:
                config, summary = run.config, run.summary  # type: ignore
            except Exception as e:
                print(e)
                continue

            if isinstance(config, str):
                config = {k: v["value"] for k, v in json.loads(config).items() if not k.startswith("_")}
            if isinstance(summary._json_dict, str):  # type: ignore
                summary = json.loads(summary._json_dict)  # type: ignore

            # Extract revision commit and branch
            try:
                commit_hash, branch = config["revision"].split("@", 1)
                commit_hashes.add(commit_hash)
            except ValueError:
                # Ignore this run if the revision has been corrupted for some unknown reason
                continue

            # Ignore runs associated with a commit that is not part of the official repository
            if not branch.startswith("Genesis-Embodied-AI/") and not self.dev_allow_all_branches:
                continue

            # Skip runs did not finish for some reason
            if run.state != "finished":
                continue

            # Do not store new records if the desired number of revision is already reached
            if len(records_by_commit_hash) == self.max_valid_revisions and commit_hash not in records_by_commit_hash:
                continue

            wandb_parser(
                benchmark_under_test=benchmark_under_test,
                records_by_commit_hash=records_by_commit_hash,
                config=config,
                summary=summary,
                commit_hash=commit_hash,
            )
        return records_by_commit_hash

    def build_table(
        self,
        config_param_names: tuple[str, ...],
        alias: str,
        metric: str,
        benchmark_run_under_test: BenchmarkRunUnderTest,
        records_by_commit_hash: dict[str, Any],
        sign: int,
    ) -> tuple[list[str], bool, bool]:
        # together these rows contain the text of the markdwon
        markdown_rows = []
        rows = []
        alert_found, reg_found = False, False

        # the labels in the header row of the table
        header_cells = (
            "status",
            *config_param_names,
            f"current {alias}",
            f"baseline {alias} [last (mean ± std)] (*1)",
            f"Δ {alias} (*2)",
        )
        header = "| " + " | ".join(header_cells) + " |"
        align = "|:------:|" + "|".join([":---" for _ in config_param_names]) + "|---:|---:|---:|"

        row_data = {}
        for config_params_fdict in sorted(
            benchmark_run_under_test.results.keys(), key=SortKey(config_param_names=config_param_names)
        ):
            value_cur = benchmark_run_under_test.results[config_params_fdict][metric]
            is_int = isinstance(value_cur, int) or value_cur.is_integer()
            value_repr = fmt_num(value_cur, is_int)

            params_repr = [config_params_fdict.get(k, "-") for k in config_param_names]
            row_data = {
                **dict(zip(config_param_names, params_repr)),
                "current": value_cur,
                "baseline_last": None,
                "baseline_mean": None,
                "baseline_min": None,
                "baseline_max": None,
                "status": None,
            }

            values_prev = [
                record[config_params_fdict][metric]
                for record in records_by_commit_hash.values()
                if config_params_fdict in record
            ]
            if values_prev:
                value_last = values_prev[0]
                value_ref = statistics.fmean(values_prev)
                delta = (value_cur - value_last) / value_last * 100.0

                row_data["baseline_last"] = int(value_last) if is_int else float(value_last)

                stats_repr = f"{fmt_num(value_last, is_int)}"
                delta_repr = f"{delta:+.1f}%"
                if len(values_prev) >= self.max_valid_revisions:
                    row_data["baseline_mean"] = int(value_ref) if is_int else float(value_ref)
                    row_data["baseline_min"] = int(min(values_prev)) if is_int else float(min(values_prev))
                    row_data["baseline_max"] = int(max(values_prev)) if is_int else float(max(values_prev))

                    value_ci95 = (
                        statistics.stdev(values_prev) / math.sqrt(len(values_prev)) * 1.96
                        if len(values_prev) > 1
                        else math.nan
                    )
                    stats_repr += f" ({fmt_num(value_ref, is_int)} ± {fmt_num(value_ci95, is_int)})"
                    if sign * delta < -self.metrics_tol[metric]:
                        row_data["status"] = "regression"

                        delta_repr = f"**{delta_repr}**"
                        picto = "🔴"
                        reg_found = True
                    elif sign * delta > self.metrics_tol[metric]:
                        row_data["status"] = "alert"

                        delta_repr = f"**{delta_repr}**"
                        picto = "⚠️"
                        alert_found = True
                    else:
                        row_data["status"] = "ok"

                        picto = "✅"
                else:
                    row_data["status"] = "n/a"

                    picto = "ℹ️"
            else:
                picto, stats_repr, delta_repr = "ℹ️", "---", "---"

            markdown_rows.append("| " + " | ".join((picto, *params_repr, value_repr, stats_repr, delta_repr)) + " |")
            rows.append(row_data)

        blist = [f"- Commit {i}: {sha}" for i, sha in enumerate(records_by_commit_hash.keys(), 1)]
        baseline_block = ["**Baselines considered:** " + f"**{len(records_by_commit_hash)}** commits"] + blist

        with self.csv_out_file_by_metric_name[metric].open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row_data.keys())
            w.writeheader()
            for rec in rows:
                w.writerow(rec)

        return [header, align] + markdown_rows + [""] + baseline_block, reg_found, alert_found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-artifacts-dir", type=str, required=True)
    parser.add_argument("--mem-artifacts-dir", type=str, required=True)
    parser.add_argument(
        "--max-valid-revisions",
        type=int,
        default=10,
        help="limits how many git commits are used to build the baseline statistics",
    )
    parser.add_argument("--max-fetch-revisions", type=int, default=10)
    parser.add_argument("--runtime-fps-regression-tolerance-pct", type=float, default=10)
    parser.add_argument("--compile-time-regression-tolerance-pct", type=float, default=10)
    parser.add_argument("--mem-regression-tolerance-pct", type=float, default=10)
    parser.add_argument("--check-body-path", type=str, required=True)
    parser.add_argument("--csv-runtime-fps-path", type=str, required=True)
    parser.add_argument("--csv-compile-time-path", type=str, required=True)
    parser.add_argument("--csv-mem-path", type=str, required=True)
    parser.add_argument("--exit-code-regression", type=int, default=42)
    parser.add_argument("--exit-code-alert", type=int, default=43)
    parser.add_argument("--dev-skip-speed", action="store_true")
    parser.add_argument("--dev-allow-all-branches", action="store_true")
    args = parser.parse_args()

    alarm = Alarm(args=args)

    results_under_test_speed = BenchmarkRunUnderTest(
        artifacts_dir=alarm.speed_artifacts_dir, metric_keys=alarm.speed_metric_keys, filename_glob="speed_test*.txt"
    )
    results_under_test_mem = BenchmarkRunUnderTest(
        artifacts_dir=alarm.mem_artifacts_dir, metric_keys=alarm.mem_metric_keys, filename_glob="mem_test*.txt"
    )

    speed_records_by_commit_hash = {}
    if not alarm.dev_skip_speed:
        speed_records_by_commit_hash = alarm.fetch_wandb_data(
            benchmark_under_test=results_under_test_speed,
            run_name_prefix="speed-",
            wandb_parser=WandbParserNewFormat(),
        )

    mem_records_by_commit_hash = alarm.fetch_wandb_data(
        benchmark_under_test=results_under_test_mem, run_name_prefix="mem-", wandb_parser=WandbParserNewFormat()
    )

    reg_found, alert_found = False, False
    table_by_metric_name: dict[str, list[str]] = {}
    reg_found, alert_found = False, False
    for metric, alias, sign, results_under_test_, records_by_commit_hash_ in (
        (alarm.metric_runtime_fps, "FPS", 1, results_under_test_speed, speed_records_by_commit_hash),
        (alarm.metric_compile_time, "compile", -1, results_under_test_speed, speed_records_by_commit_hash),
        (alarm.metric_max_mem_mb, "memory", -1, results_under_test_mem, mem_records_by_commit_hash),
    ):
        (table_by_metric_name[metric], reg_found_, alert_found_) = alarm.build_table(
            config_param_names=results_under_test_.config_param_names,
            alias=alias,
            metric=metric,
            sign=sign,
            benchmark_run_under_test=results_under_test_,
            records_by_commit_hash=records_by_commit_hash_,
        )
        reg_found |= reg_found_
        alert_found |= alert_found_

    thr_repr = ", ".join(
        f"{alias} ± {alarm.metrics_tol[metric]:.0f}%"
        for metric, alias in (
            (alarm.metric_runtime_fps, "runtime"),
            (alarm.metric_compile_time, "compile"),
            (alarm.metric_max_mem_mb, "mem"),
        )
    )

    check_body = "\n".join(
        [
            f"Thresholds: {thr_repr}",
            "",
            "### Runtime FPS",
            *table_by_metric_name[alarm.metric_runtime_fps],
            "",
            "### Compile Time",
            *table_by_metric_name[alarm.metric_compile_time],
            "",
            "### Memory usage",
            *table_by_metric_name[alarm.metric_max_mem_mb],
            "",
            f"- (*1) last: last commit on main, mean/std: stats over commit hashes {alarm.max_valid_revisions} commits if available.",
            "- (*2) Δ: relative difference between PR and last commit on main, i.e. (PR - main) / main * 100%.",
        ]
    )

    alarm.check_body_path.write_text(check_body + "\n", encoding="utf-8")

    if reg_found:
        sys.exit(int(args.exit_code_regression))

    if alert_found:
        sys.exit(int(args.exit_code_alert))

    sys.exit(0)
