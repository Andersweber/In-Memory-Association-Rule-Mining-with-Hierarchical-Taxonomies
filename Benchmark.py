"""
Benchmark.py — Combined benchmark for GoWish thesis tables.

Runs five experiments with one command, each with its own flags:

  Experiment 1: Sensitivity sweep (τ/λ grid)            → Table 8
  Experiment 2: Basic vs. Python Cumulate                → Table 9
  Experiment 3: Example rules at K=5, s=0.02             → Table 3
  Experiment 4: K-sweep (K=1..5, all implementations)   → Figures 2,3,4,8 + Tables 5,10
  Experiment 5: Support sweep (s varies, K=3 fixed)      → Figures 5,6 + Table 11

Usage:
    python Benchmark.py <merged_parquet_dir> [options]

Example (paper settings, all experiments):
    python Benchmark.py Data/samples/100000 \\
        --repeats 3 --output-dir benchmark_results

Skip experiments with --skip:
    python Benchmark.py Data/samples/100000 --skip sensitivity example_rules
"""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from itertools import product as itertools_product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------

REPO_ROOT             = Path(__file__).resolve().parent
CUMULATE_PY_DIR       = REPO_ROOT / "apriori_cumulate" / "python"
MLX_ANCESTOR_DIR      = REPO_ROOT / "apriori_mlx_ancestor"
MLX_FLAT_DIR          = REPO_ROOT / "apriori_mlx_tend"
CUMULATE_PIPELINE     = CUMULATE_PY_DIR / "Apriori_Cumulate_Python.py"
MLX_ANCESTOR_PIPELINE = MLX_ANCESTOR_DIR / "Apriori_MLX_Ancestor.py"
MLX_FLAT_PIPELINE     = MLX_FLAT_DIR / "Apriori_MLX_Flat.py"

# ---------------------------------------------------------------------------
# Import helpers from cumulate ProductionPipeline (used in-process by sweep)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(CUMULATE_PY_DIR))
from Apriori_Cumulate_Python import (       # noqa: E402
    now_stamp, format_elapsed, timed_step,
    load_data, clean_data, build_transactions, encode_transactions,
    build_ancestors_from_tokens, mine_rules_raw,
    add_score_and_rank, dedupe_family, dedupe_antimirror, pretty_rules,
    label_of, level_of, branch_of,
)
from frequent_patterns import fpcommon as fpc  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combined benchmark for GoWish thesis (Tables 3, 8, 9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "base",
        help="Directory containing pre-merged parquet(s) "
             "(columns: wishlist_id, category_name).",
    )
    p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Root directory for all benchmark outputs.",
    )
    p.add_argument(
        "--skip", nargs="*",
        choices=["sensitivity", "basic_vs_cumulate", "example_rules",
                 "k_sweep", "support_sweep"],
        default=[],
        help="Experiments to skip. By default all five run.",
    )
    p.add_argument(
        "--repeats", type=int, default=1,
        help="Timed repeats for the Basic vs. Cumulate subprocess runs.",
    )
    p.add_argument(
        "--python-exe", default=sys.executable,
        help="Python interpreter for subprocesses.",
    )
    p.add_argument(
        "--sample", type=int, default=None,
        help="Restrict to the first N wishlists (quick smoke-test).",
    )

    # ── Experiment 1: sensitivity sweep ─────────────────────────────────────
    g1 = p.add_argument_group("Experiment 1 — Sensitivity sweep (Table 8)")
    g1.add_argument("--sweep-k",       type=int,   default=3,   metavar="K")
    g1.add_argument("--sweep-support", type=float, default=0.02, metavar="S")
    g1.add_argument("--sweep-max-ante-len", type=int, default=3, metavar="A")
    g1.add_argument("--sweep-max-cons-len", type=int, default=2, metavar="B")
    g1.add_argument(
        "--sweep-tau", nargs="+", type=float,
        default=[0.5, 0.6, 0.7], metavar="T",
        help="Confidence thresholds τ to sweep.",
    )
    g1.add_argument(
        "--sweep-lambda", nargs="+", type=float,
        default=[1.2, 1.5, 2.0], metavar="L",
        help="Lift floors λ to sweep.",
    )

    # ── Experiment 2: Basic vs. Cumulate ────────────────────────────────────
    g2 = p.add_argument_group("Experiment 2 — Basic vs. Cumulate (Table 9)")
    g2.add_argument("--basic-k",       type=int,   default=3,    metavar="K")
    g2.add_argument("--basic-support", type=float, default=0.02, metavar="S")
    g2.add_argument("--basic-conf",    type=float, default=0.6,  metavar="C")
    g2.add_argument("--basic-lift",    type=float, default=1.5,  metavar="L")
    g2.add_argument("--basic-max-len", type=int,   default=3,    metavar="M")

    # ── Experiment 3: example rules ──────────────────────────────────────────
    g3 = p.add_argument_group("Experiment 3 — Example rules (Table 3)")
    g3.add_argument("--example-k",           type=int,   default=5,    metavar="K")
    g3.add_argument("--example-support",     type=float, default=0.02, metavar="S")
    g3.add_argument("--example-conf",        type=float, default=0.6,  metavar="C")
    g3.add_argument("--example-lift",        type=float, default=1.5,  metavar="L")
    g3.add_argument("--example-max-ante-len", type=int,  default=3,    metavar="A")
    g3.add_argument("--example-max-cons-len", type=int,  default=2,    metavar="B")
    g3.add_argument("--example-top-n",       type=int,  default=10,
                    help="Number of top rules to print for Table 3.")

    # ── Experiment 4: K-sweep ────────────────────────────────────────────────
    g4 = p.add_argument_group(
        "Experiment 4 — K-sweep (Figures 2,3,4,8 + Tables 5,10)")
    g4.add_argument(
        "--ksweep-k", nargs="+", type=int, default=[1, 2, 3, 4, 5], metavar="K",
        help="K values to sweep.",
    )
    g4.add_argument("--ksweep-support", type=float, default=0.02, metavar="S")
    g4.add_argument("--ksweep-conf",    type=float, default=0.6,  metavar="C")
    g4.add_argument("--ksweep-lift",    type=float, default=1.5,  metavar="L")
    g4.add_argument("--ksweep-max-len", type=int,   default=5,    metavar="M")

    # ── Experiment 5: support sweep ──────────────────────────────────────────
    g5 = p.add_argument_group(
        "Experiment 5 — Support sweep (Figures 5,6 + Table 11)")
    g5.add_argument(
        "--ssweep-support", nargs="+", type=float,
        default=[0.05, 0.03, 0.02, 0.01], metavar="S",
        help="Support values to sweep.",
    )
    g5.add_argument("--ssweep-k",       type=int,   default=3,   metavar="K")
    g5.add_argument("--ssweep-conf",    type=float, default=0.6, metavar="C")
    g5.add_argument("--ssweep-lift",    type=float, default=1.5, metavar="L")
    g5.add_argument("--ssweep-max-len", type=int,   default=5,   metavar="M")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared subprocess helpers
# ---------------------------------------------------------------------------

def _parse_duration(value: str) -> Optional[float]:
    value = value.strip().replace(",", ".")
    m = re.fullmatch(r"(?:(\d+)m\s*)?([0-9.]+)s", value)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    try:
        return float(value.rstrip("s"))
    except ValueError:
        return None


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _first_duration(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return _parse_duration(m.group(1)) if m else None


def _first_int(text: str, patterns: List[str]) -> Optional[int]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return _parse_int(m.group(1))
    return None


def extract_metrics(stdout: str, stderr: str) -> Dict[str, Any]:
    text = stdout + "\n" + stderr
    out: Dict[str, Any] = {
        "load_data_seconds": _first_duration(
            text, r"Loading data done in ([0-9.m\s]+s)"),
        "clean_data_seconds": _first_duration(
            text, r"Cleaning data done in ([0-9.m\s]+s)"),
        "build_transactions_seconds": _first_duration(
            text, r"Building transactions done in ([0-9.m\s]+s)"),
        "encode_transactions_seconds": _first_duration(
            text, r"(?:Encoding transactions|Encoding \(TransactionEncoder\)) done in ([0-9.m\s]+s)"),
        "build_ancestor_map_seconds": _first_duration(
            text, r"Building ancestor map done in ([0-9.m\s]+s)"),
        "apriori_seconds": _first_duration(
            text, r"(?:Apriori mining|Apriori) done in ([0-9.m\s]+s)"),
        "association_rules_seconds": _first_duration(
            text, r"(?:Association rule generation|Association rules) done in ([0-9.m\s]+s)"),
        "postprocess_seconds": _first_duration(
            text, r"Scoring & deduplicating done in ([0-9.m\s]+s)"),
        "transactions": _first_int(
            text, [r"Total baskets\s*:\s*([0-9,]+)", r"baskets:\s*([0-9,]+)"]),
        "unique_tokens": _first_int(
            text, [r"Unique items\s*:\s*([0-9,]+)", r"Vocabulary:\s*([0-9,]+)"]),
        "frequent_itemsets": _first_int(
            text, [r"Frequent itemsets:\s*([0-9,]+)"]),
        "raw_rules": _first_int(
            text, [r"Raw rules:\s*([0-9,]+)", r"Rules:\s*([0-9,]+)"]),
        "score_rank_rules": _first_int(
            text, [r"After score\+rank:\s*([0-9,]+)\s+rules"]),
        "family_dedupe_rules": _first_int(
            text, [r"After family dedupe:\s*([0-9,]+)\s+rules"]),
        "antimirror_dedupe_rules": _first_int(
            text, [r"After antimirror dedupe:\s*([0-9,]+)\s+rules"]),
        "final_rules": _first_int(
            text, [r"Found\s+([0-9,]+)\s+rules"]),
    }
    core = [out.get("apriori_seconds"), out.get("association_rules_seconds")]
    out["algorithm_core_seconds"] = sum(float(v) for v in core if v is not None)
    return out


def run_subprocess(command: List[str], run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = subprocess.run(
        command, cwd=str(REPO_ROOT), capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    elapsed = time.perf_counter() - t0
    (run_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    metrics = extract_metrics(result.stdout, result.stderr)
    metrics["runtime_seconds"] = elapsed
    metrics["success"]         = result.returncode == 0
    metrics["return_code"]     = result.returncode
    metrics["command"]         = " ".join(shlex.quote(part) for part in command)
    return metrics


def _annotate_rules(rules: pd.DataFrame) -> pd.DataFrame:
    r = rules.copy()
    r["a_toks"]     = r["antecedents"].apply(lambda s: tuple(sorted(s)))
    r["b_toks"]     = r["consequents"].apply(lambda s: tuple(sorted(s)))
    r["a_lbl"]      = r["a_toks"].apply(lambda t: " + ".join(label_of(x) for x in t))
    r["b_lbl"]      = r["b_toks"].apply(lambda t: " + ".join(label_of(x) for x in t))
    r["a_levels"]   = r["a_toks"].apply(lambda t: tuple(level_of(x) for x in t))
    r["b_levels"]   = r["b_toks"].apply(lambda t: tuple(level_of(x) for x in t))
    r["a_branches"] = r["a_toks"].apply(lambda t: tuple(branch_of(x) for x in t))
    r["b_branches"] = r["b_toks"].apply(lambda t: tuple(branch_of(x) for x in t))
    return r


# ===========================================================================
# EXPERIMENT 1 — Sensitivity sweep (Table 8)
# ===========================================================================

def run_sensitivity_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    max_len = args.sweep_max_ante_len + args.sweep_max_cons_len

    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Sensitivity sweep  (Table 8)")
    print(f"  K={args.sweep_k}  s={args.sweep_support}  "
          f"max_ante={args.sweep_max_ante_len}  max_cons={args.sweep_max_cons_len}")
    print(f"  tau    ∈ {args.sweep_tau}")
    print(f"  lambda ∈ {args.sweep_lambda}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    # ── Load & prepare (once) ────────────────────────────────────────────────
    with timed_step("Loading data"):
        df = load_data(args.base)

    with timed_step("Cleaning data"):
        df_min = clean_data(df)

    if args.sample:
        wishlists = df_min["wishlist_id"].unique()[:args.sample]
        df_min = df_min[df_min["wishlist_id"].isin(wishlists)]
        print(f"  Sampled to {df_min['wishlist_id'].nunique():,} wishlists")

    with timed_step("Building transactions"):
        transactions, all_tokens = build_transactions(df_min, args.sweep_k)
        path_map = {tok: label_of(tok) for tok in all_tokens}
        print(f"  Transactions: {len(transactions):,}  Vocabulary: {len(all_tokens):,}")

    with timed_step("Encoding transactions"):
        df_encoded = encode_transactions(transactions, all_tokens)

    with timed_step("Building ancestor map"):
        branch_ancestry = fpc.build_branch_ancestry(df_min, name_col="category_name")
        ancestors = build_ancestors_from_tokens(all_tokens, branch_ancestry=branch_ancestry)

    # ── Run Apriori ONCE at the lowest tau ───────────────────────────────────
    min_tau = min(args.sweep_tau)
    print(f"\n[{now_stamp()}] Running Apriori once at min_conf={min_tau} …")
    t_ap = time.perf_counter()
    rules_raw = mine_rules_raw(
        df_encoded,
        min_support=args.sweep_support,
        min_conf=min_tau,
        max_len=max_len,
        ancestors=ancestors,
        max_ante_len=args.sweep_max_ante_len,
        max_cons_len=args.sweep_max_cons_len,
        require_single_consequent=False,
        path_map=path_map,
        branch_ancestry=branch_ancestry,
        candidate_pruning="ancestor-same-path",
        rule_filtering="full",
    )
    print(f"[{now_stamp()}] Apriori done in {format_elapsed(time.perf_counter() - t_ap)}")
    print(f"  Raw rules at tau={min_tau}: {len(rules_raw):,}\n")

    rules_annotated = _annotate_rules(rules_raw)

    # ── Post-process each (tau, lambda) in memory ────────────────────────────
    summary_rows = []
    for tau, lam in itertools_product(args.sweep_tau, args.sweep_lambda):
        label = f"tau{tau}_lambda{lam}".replace(".", "p")
        print(f"  tau={tau}  lambda={lam} …", end=" ", flush=True)
        t0 = time.perf_counter()

        r = rules_annotated[rules_annotated["confidence"] >= tau].copy()
        r = add_score_and_rank(r, args.sweep_support, lam)
        n_score  = len(r)
        r = dedupe_family(r)
        n_family = len(r)
        r = dedupe_antimirror(r)
        n_final  = len(r)

        print(f"→ {n_final:,} rules  ({time.perf_counter()-t0:.3f}s)")
        pretty_rules(r).to_csv(out_dir / f"rules_{label}.csv", index=False)
        summary_rows.append({
            "tau": tau, "lambda": lam,
            "raw_rules": len(rules_raw),
            "after_conf_lift_filter": n_score,
            "after_family_dedupe": n_family,
            "final_rules": n_final,
        })

    # ── Save & print ─────────────────────────────────────────────────────────
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSensitivity grid (final rules):")
    print(f"{'':>6}", end="")
    for lam in args.sweep_lambda:
        print(f"   λ={lam}", end="")
    print()
    for tau in args.sweep_tau:
        print(f"τ={tau}", end="")
        for lam in args.sweep_lambda:
            val = next(r["final_rules"] for r in summary_rows
                       if r["tau"] == tau and r["lambda"] == lam)
            print(f"  {val:>7,}", end="")
        print()

    print(
        f"\n[{now_stamp()}] Sensitivity sweep done in "
        f"{format_elapsed(time.perf_counter() - total_start)}"
    )
    print(f"Results saved to {out_dir}/")


# ===========================================================================
# EXPERIMENT 2 — Basic vs. Python Cumulate (Table 9)
# ===========================================================================

def run_basic_vs_cumulate(args: argparse.Namespace, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Basic vs. Python Cumulate  (Table 9)")
    print(f"  K={args.basic_k}  s={args.basic_support}  "
          f"conf={args.basic_conf}  lift={args.basic_lift}  max_len={args.basic_max_len}")
    print(f"  repeats={args.repeats}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for repeat_id in range(1, args.repeats + 1):
        configs.append({
            "implementation": "python_cumulate",
            "repeat_id": repeat_id,
            "script": str(CUMULATE_PIPELINE),
            "extra_args": ["--candidate-pruning", "none", "--rule-filtering", "none"],
            "rules_filename": "rules_python.csv",
        })
        configs.append({
            "implementation": "mlxtend_basic",
            "repeat_id": repeat_id,
            "script": str(MLX_ANCESTOR_PIPELINE),
            "extra_args": ["--mode", "basic"],
            "rules_filename": "rules_mlxtend.csv",
        })

    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        impl      = cfg["implementation"]
        repeat_id = cfg["repeat_id"]
        run_id    = (
            f"{impl}_k{args.basic_k}"
            f"_s{str(args.basic_support).replace('.','p')}"
            f"_r{repeat_id}"
        )
        run_dir    = out_dir / "runs" / run_id
        rules_file = run_dir / cfg["rules_filename"]

        command = [
            args.python_exe, cfg["script"], args.base,
            "--k-levels",    str(args.basic_k),
            "--min-support", str(args.basic_support),
            "--min-conf",    str(args.basic_conf),
            "--min-lift",    str(args.basic_lift),
            "--max-len",     str(args.basic_max_len),
            "--output",      str(rules_file),
        ] + cfg["extra_args"]

        print(f"Running {impl} (repeat {repeat_id}) …", flush=True)
        metrics = run_subprocess(command, run_dir)
        metrics["implementation"] = impl
        metrics["repeat_id"]      = repeat_id

        status = "OK" if metrics["success"] else f"FAILED (rc={metrics['return_code']})"
        print(
            f"  {status}  wall={metrics['runtime_seconds']:.1f}s  "
            f"apriori={metrics.get('apriori_seconds') or 0:.3f}s  "
            f"rules={metrics.get('final_rules') or metrics.get('raw_rules')}"
        )
        if not metrics["success"]:
            print(f"  stderr → {run_dir / 'stderr.log'}")
        rows.append(metrics)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    summary_path = out_dir / "comparison_summary.csv"
    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved comparison summary to {summary_path}")

    print(f"\n{'Implementation':<28} {'Rep':>3} {'Wall (s)':>10} "
          f"{'Apriori (s)':>12} {'Rules':>8}")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['implementation']:<28} "
            f"{row['repeat_id']:>3} "
            f"{row['runtime_seconds']:>10.2f} "
            f"{row.get('apriori_seconds') or 0:>12.3f} "
            f"{row.get('final_rules') or row.get('raw_rules') or '?':>8}"
        )


# ===========================================================================
# EXPERIMENT 3 — Example rules (Table 3)
# ===========================================================================

def run_example_rules(args: argparse.Namespace, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Example rules  (Table 3)")
    print(f"  K={args.example_k}  s={args.example_support}  "
          f"conf={args.example_conf}  lift={args.example_lift}")
    print(f"  max_ante={args.example_max_ante_len}  max_cons={args.example_max_cons_len}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir    = out_dir / "runs" / f"k{args.example_k}_s{str(args.example_support).replace('.','p')}"
    rules_file = out_dir / f"rules_k{args.example_k}_s{str(args.example_support).replace('.','p')}.csv"

    command = [
        args.python_exe, str(CUMULATE_PIPELINE), args.base,
        "--k-levels",     str(args.example_k),
        "--min-support",  str(args.example_support),
        "--min-conf",     str(args.example_conf),
        "--min-lift",     str(args.example_lift),
        "--max-ante-len", str(args.example_max_ante_len),
        "--max-cons-len", str(args.example_max_cons_len),
        "--output",       str(rules_file),
    ]

    print(
        f"Running Python Cumulate  K={args.example_k}  s={args.example_support} …",
        flush=True,
    )
    metrics = run_subprocess(command, run_dir)

    status = "OK" if metrics["success"] else f"FAILED (rc={metrics['return_code']})"
    print(
        f"  {status}  wall={metrics['runtime_seconds']:.1f}s  "
        f"rules={metrics.get('final_rules')}"
    )
    if not metrics["success"]:
        print(f"  stderr → {run_dir / 'stderr.log'}")
        return

    if rules_file.exists():
        rules_df = pd.read_csv(rules_file)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.width", 220)
        top = rules_df.sort_values("lift", ascending=False).head(args.example_top_n)
        print(f"\nTop {args.example_top_n} rules by lift "
              f"(K={args.example_k}, s={args.example_support}):\n")
        print(top[["rule", "support", "confidence", "lift"]].to_string(index=False))
        print(f"\nAll {len(rules_df):,} rules saved to {rules_file}")


# ===========================================================================
# EXPERIMENT 4 — K-sweep (Figures 2, 3, 4, 8 + Tables 5, 10)
# ===========================================================================

def run_k_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: K-sweep  (Figures 2,3,4,8 + Tables 5,10)")
    print(f"  K ∈ {args.ksweep_k}  s={args.ksweep_support}  "
          f"conf={args.ksweep_conf}  lift={args.ksweep_lift}")
    print(f"  repeats={args.repeats}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    # python_cumulate and mlxtend_flat at each K
    implementations = [
        ("python_cumulate", str(CUMULATE_PIPELINE), [
            "--max-ante-len", str(args.ksweep_max_len),
            "--max-cons-len", str(args.ksweep_max_len),
        ]),
        ("mlxtend_flat", str(MLX_FLAT_PIPELINE), [
            "--max-len", str(args.ksweep_max_len),
        ]),
    ]

    rows: List[Dict[str, Any]] = []
    for k in args.ksweep_k:
        for repeat_id in range(1, args.repeats + 1):
            for impl, script, extra_args in implementations:
                run_id    = f"{impl}_k{k}_r{repeat_id}"
                run_dir   = out_dir / "runs" / run_id
                rules_file = run_dir / "rules.csv"

                # mlxtend_flat has no --k-levels flag; use --level 0 (flat mode)
                if impl == "mlxtend_flat":
                    command = [
                        args.python_exe, script, args.base,
                        "--min-support", str(args.ksweep_support),
                        "--min-conf",    str(args.ksweep_conf),
                        "--min-lift",    str(args.ksweep_lift),
                        "--output",      str(rules_file),
                    ] + extra_args
                else:
                    command = [
                        args.python_exe, script, args.base,
                        "--k-levels",    str(k),
                        "--min-support", str(args.ksweep_support),
                        "--min-conf",    str(args.ksweep_conf),
                        "--min-lift",    str(args.ksweep_lift),
                        "--output",      str(rules_file),
                    ] + extra_args

                print(f"  {impl}  K={k}  repeat={repeat_id} …", end=" ", flush=True)
                metrics = run_subprocess(command, run_dir)
                metrics.update({
                    "implementation": impl,
                    "k_levels":       k,
                    "repeat_id":      repeat_id,
                    "min_support":    args.ksweep_support,
                    "experiment":     "k_sweep",
                })

                status = "OK" if metrics["success"] else f"FAILED (rc={metrics['return_code']})"
                print(
                    f"{status}  wall={metrics['runtime_seconds']:.1f}s  "
                    f"apriori={metrics.get('apriori_seconds') or 0:.3f}s  "
                    f"rules={metrics.get('final_rules') or metrics.get('raw_rules')}"
                )
                if not metrics["success"]:
                    print(f"    stderr → {run_dir / 'stderr.log'}")
                rows.append(metrics)

    _save_csv(rows, out_dir / "k_sweep_summary.csv")
    _print_sweep_table(rows, group_col="k_levels", group_label="K")


# ===========================================================================
# EXPERIMENT 5 — Support sweep (Figures 5, 6 + Table 11)
# ===========================================================================

def run_support_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENT 5: Support sweep  (Figures 5,6 + Table 11)")
    print(f"  K={args.ssweep_k}  s ∈ {args.ssweep_support}  "
          f"conf={args.ssweep_conf}  lift={args.ssweep_lift}")
    print(f"  repeats={args.repeats}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for s in args.ssweep_support:
        for repeat_id in range(1, args.repeats + 1):
            s_str     = str(s).replace(".", "p")
            run_id    = f"python_cumulate_s{s_str}_r{repeat_id}"
            run_dir   = out_dir / "runs" / run_id
            rules_file = run_dir / "rules.csv"

            command = [
                args.python_exe, str(CUMULATE_PIPELINE), args.base,
                "--k-levels",     str(args.ssweep_k),
                "--min-support",  str(s),
                "--min-conf",     str(args.ssweep_conf),
                "--min-lift",     str(args.ssweep_lift),
                "--max-ante-len", str(args.ssweep_max_len),
                "--max-cons-len", str(args.ssweep_max_len),
                "--output",       str(rules_file),
            ]

            print(f"  python_cumulate  s={s}  repeat={repeat_id} …", end=" ", flush=True)
            metrics = run_subprocess(command, run_dir)
            metrics.update({
                "implementation": "python_cumulate",
                "k_levels":       args.ssweep_k,
                "min_support":    s,
                "repeat_id":      repeat_id,
                "experiment":     "support_sweep",
            })

            status = "OK" if metrics["success"] else f"FAILED (rc={metrics['return_code']})"
            print(
                f"{status}  wall={metrics['runtime_seconds']:.1f}s  "
                f"apriori={metrics.get('apriori_seconds') or 0:.3f}s  "
                f"rules={metrics.get('final_rules') or metrics.get('raw_rules')}"
            )
            if not metrics["success"]:
                print(f"    stderr → {run_dir / 'stderr.log'}")
            rows.append(metrics)

    _save_csv(rows, out_dir / "support_sweep_summary.csv")
    _print_sweep_table(rows, group_col="min_support", group_label="support")


# ---------------------------------------------------------------------------
# Shared sweep helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary to {path}")


def _print_sweep_table(
    rows: List[Dict[str, Any]],
    group_col: str,
    group_label: str,
) -> None:
    print(f"\n{'Implementation':<25} {group_label:>10} {'Rep':>3} "
          f"{'Wall (s)':>10} {'Apriori (s)':>12} {'Rules':>8}")
    print("-" * 73)
    for row in rows:
        print(
            f"{row['implementation']:<25} "
            f"{row[group_col]:>10} "
            f"{row['repeat_id']:>3} "
            f"{row['runtime_seconds']:>10.2f} "
            f"{row.get('apriori_seconds') or 0:>12.3f} "
            f"{row.get('final_rules') or row.get('raw_rules') or '?':>8}"
        )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    skip = set(args.skip or [])
    run  = {
        "sensitivity", "basic_vs_cumulate", "example_rules",
        "k_sweep", "support_sweep",
    } - skip

    out_root = Path(args.output_dir).resolve()

    print(f"[{now_stamp()}] Benchmark started")
    print(f"Data path    : {args.base}")
    print(f"Output dir   : {out_root}")
    print(f"Running      : {', '.join(sorted(run))}")
    if skip:
        print(f"Skipping     : {', '.join(sorted(skip))}")
    if args.sample:
        print(f"Sample       : {args.sample} wishlists")
    print()

    if "sensitivity" in run:
        run_sensitivity_sweep(args, out_root / "sensitivity")

    if "basic_vs_cumulate" in run:
        run_basic_vs_cumulate(args, out_root / "basic_vs_cumulate")

    if "example_rules" in run:
        run_example_rules(args, out_root / "example_rules")

    if "k_sweep" in run:
        run_k_sweep(args, out_root / "k_sweep")

    if "support_sweep" in run:
        run_support_sweep(args, out_root / "support_sweep")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[{now_stamp()}] Benchmark finished in {format_elapsed(total_elapsed)}")
    print(f"All results under {out_root}/")


if __name__ == "__main__":
    main()
