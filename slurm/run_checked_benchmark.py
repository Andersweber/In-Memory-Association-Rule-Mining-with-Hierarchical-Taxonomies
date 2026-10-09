#!/usr/bin/env python3
"""Run Benchmark.py with mandatory C++, checked outputs and child RSS accounting.

The mining algorithms and experiment parameters stay in Benchmark.py. This
launcher replaces only its subprocess helper and checks experiment outputs.
Run on Linux via run_slurm_benchmark.sh; all Benchmark.py arguments are accepted.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter


EXPERIMENTS = {
    "sensitivity": "run_sensitivity_sweep",
    "basic_vs_cumulate": "run_basic_vs_cumulate",
    "example_rules": "run_example_rules",
    "k_sweep": "run_k_sweep",
    "support_sweep": "run_support_sweep",
    "l0_pair_example": "run_l0_pair_example",
    "rule_candidate_space": "run_rule_candidate_space",
    "held_out_recall": "run_held_out_recall",
    "full_dataset": "run_full_dataset_scalability",
}
IMPLEMENTATIONS = ("python_cumulate", "mlxtend_flat", "cpp_cumulate")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def rss_mib(usage):
    return usage.ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)


def run_child(command, cwd, run_dir, audit_dir, extract_metrics):
    """wait4 records this child's exit status and peak RSS, including failures."""
    command = [str(part) for part in command]
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "command": command, "run_dir": str(run_dir)}
    write_json(audit_dir / "process.json", record)
    started = time.perf_counter()
    peak = None
    with (audit_dir / "stdout.log").open("wb") as stdout, (audit_dir / "stderr.log").open("wb") as stderr:
        try:
            with subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr) as child:
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                return_code = child.returncode
                peak = rss_mib(usage)
        except OSError as exc:
            return_code = 127
            stderr.write((str(exc) + "\n").encode())
    elapsed = time.perf_counter() - started
    stdout_text = (audit_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
    stderr_text = (audit_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
    # Keep the original log locations too. Audit logs preserve every warmup,
    # including support-sweep warmups which otherwise overwrite the same files.
    for name in ("stdout.log", "stderr.log"):
        shutil.copyfile(audit_dir / name, run_dir / name)
    record.update(status="success" if return_code == 0 else "failed",
                  return_code=return_code, runtime_seconds=elapsed, peak_rss_mib=peak)
    write_json(audit_dir / "process.json", record)
    with (audit_dir.parent / "subruns.jsonl").open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record) + "\n")
    if return_code != 0:
        raise RuntimeError(f"Subprocess failed with rc={return_code}: {shlex.join(command)}; logs: {audit_dir}")
    metrics = extract_metrics(stdout_text, stderr_text)
    metrics.update(success=True, return_code=return_code, runtime_seconds=elapsed,
                   peak_rss_mib=peak, command=shlex.join(command))
    return metrics


def csv_rows(path, allow_empty=False):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty output: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Missing CSV header: {path}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise RuntimeError(f"No result rows: {path}")
    return rows


def check_matrix(path, expected, fields):
    rows = csv_rows(path)
    actual = []
    for row in rows:
        if row.get("success", "").lower() != "true" or int(row["return_code"]) != 0:
            raise RuntimeError(f"Failed result row in {path}: {row}")
        actual.append(tuple(convert(row[name]) for name, convert in fields))
    if Counter(actual) != Counter(expected):
        raise RuntimeError(f"Missing, duplicate or unexpected runs in {path}: expected {expected}, got {actual}")
    return rows


def validate_outputs(args, out, selected):
    counts = {}
    repeats = range(1, args.repeats + 1)
    if "sensitivity" in selected:
        rows = json.loads((out / "sensitivity/summary.json").read_text())
        expected = Counter(itertools.product(args.sweep_tau, args.sweep_lambda))
        if Counter((r["tau"], r["lambda"]) for r in rows) != expected:
            raise RuntimeError("Incomplete sensitivity grid")
    if "basic_vs_cumulate" in selected:
        check_matrix(out / "basic_vs_cumulate/comparison_summary.csv",
                     list(itertools.product(("python_cumulate", "mlxtend_basic"), repeats)),
                     (("implementation", str), ("repeat_id", int)))
    if "example_rules" in selected:
        name = f"rules_k{args.example_k}_s{str(args.example_support).replace('.', 'p')}.csv"
        csv_rows(out / "example_rules" / name, allow_empty=True)
    for experiment, values, field, convert, filename in (
        ("k_sweep", args.ksweep_k, "k_levels", int, "k_sweep_summary.csv"),
        ("support_sweep", args.ssweep_support, "min_support", float, "support_sweep_summary.csv"),
    ):
        if experiment in selected:
            rows = check_matrix(out / experiment / filename,
                                list(itertools.product(IMPLEMENTATIONS, values, repeats)),
                                (("implementation", str), (field, convert), ("repeat_id", int)))
            counts[experiment] = dict(Counter(r["implementation"] for r in rows))
    required = {
        "l0_pair_example": ("l0_pair_example.csv",),
        "rule_candidate_space": ("summary.csv", "category_inventory.csv", "rule_candidate_space_reduction.csv"),
        "held_out_recall": ("recall_reduction_tradeoff.csv", "held_out_generalisation.csv"),
    }
    for experiment, files in required.items():
        if experiment in selected:
            for filename in files:
                csv_rows(out / experiment / filename)
    if "full_dataset" in selected:
        check_matrix(out / "full_dataset/scalability_summary.csv",
                     [("python_cumulate",), ("cpp_cumulate",)], (("implementation", str),))
    return counts


def inspect_parquet(path, required):
    import pyarrow.parquet as pq
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No Parquet files at {path}")
    rows = 0
    for file in files:
        parquet = pq.ParquetFile(file)
        missing = set(required) - set(parquet.schema_arrow.names)
        if missing:
            raise RuntimeError(f"Missing columns {sorted(missing)} in {file}")
        rows += parquet.metadata.num_rows
    if rows == 0:
        raise RuntimeError(f"Empty dataset: {path}")
    return {"path": str(path), "files": len(files), "rows": rows, "required_columns": sorted(required)}


def main():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run the benchmark on Hendrix/Linux")
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import Benchmark as benchmark
    args = benchmark.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    selected = set(EXPERIMENTS) - set(args.skip or [])
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checked_run.json").exists() or any((out / name).exists() for name in EXPERIMENTS):
        raise RuntimeError(f"Refusing to reuse benchmark outputs: {out}")
    report = {"status": "running", "expected_experiments": sorted(selected), "completed_experiments": []}
    report_path = out / "checked_run.json"
    write_json(report_path, report)
    records = []
    try:
        cpp = Path(args.cpp_exe or root / "apriori_cumulate/cpp/apriori_cumulate_cpp").resolve()
        if not cpp.is_file() or not os.access(cpp, os.X_OK):
            raise RuntimeError(f"C++ executable missing: {cpp}")
        with cpp.open("rb") as handle:
            if handle.read(4) != b"\x7fELF":
                raise RuntimeError(f"C++ binary is not Linux ELF: {cpp}")
        # Keep the parser used by main() on the verified absolute path.
        args.cpp_exe = str(cpp)
        benchmark.parse_args = lambda: args
        data = {"sample": inspect_parquet(args.mining_base or args.base, {"wishlist_id", "category_name"})}
        if selected & {"l0_pair_example", "rule_candidate_space", "held_out_recall", "full_dataset"}:
            data["catalogue"] = inspect_parquet(args.catalogue_base or args.base,
                                                {"wishlist_id", "product_id", "category_name"})
        versions = {name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "pyarrow", "scipy", "joblib", "scikit-learn", "mlxtend", "matplotlib")}
        source_files = [root / "Benchmark.py", Path(__file__).resolve(), cpp]
        for directory in ("apriori_cumulate", "apriori_mlx_ancestor", "apriori_mlx_tend"):
            source_files.extend(p for p in (root / directory).rglob("*")
                                if p.is_file() and (p.suffix in {".py", ".cpp", ".hpp"} or p.name == "Makefile"))
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        write_json(out / "provenance.json", {"arguments": vars(args).copy(), "python": sys.executable,
                                            "versions": versions, "data": data, "sha256": hashes,
                                            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                            "node": os.uname().nodename})
        print("DATA_CHECK:", json.dumps(data), flush=True)
        sequence = itertools.count(1)

        def checked_child(command, run_dir):
            audit = out / "_audit" / "subruns" / f"{next(sequence):04d}"
            metrics = run_child(command, str(root), Path(run_dir), audit, benchmark.extract_metrics)
            records.append({"command": metrics["command"], "peak_rss_mib": metrics["peak_rss_mib"]})
            return metrics

        benchmark.run_subprocess = checked_child

        def wrap_experiment(name, function):
            @functools.wraps(function)
            def checked(*positional, **keywords):
                report["active_experiment"] = name
                write_json(report_path, report)
                result = function(*positional, **keywords)
                report["completed_experiments"].append(name)
                write_json(report_path, report)
                return result
            return checked

        for name, function_name in EXPERIMENTS.items():
            setattr(benchmark, function_name, wrap_experiment(name, getattr(benchmark, function_name)))
        benchmark.main()
        if set(report["completed_experiments"]) != selected:
            raise RuntimeError("Some requested experiments were skipped")
        report["run_counts"] = validate_outputs(args, out, selected)
        report["status"] = "success"
        report.pop("active_experiment", None)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["driver_peak_rss_mib"] = rss_mib(resource.getrusage(resource.RUSAGE_SELF))
        report["successful_subprocesses_including_warmups"] = len(records)
        report["largest_successful_child_peak_rss_mib"] = max((r["peak_rss_mib"] or 0 for r in records), default=0)
        write_json(report_path, report)
    print("CHECKED_BENCHMARK_OK:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
