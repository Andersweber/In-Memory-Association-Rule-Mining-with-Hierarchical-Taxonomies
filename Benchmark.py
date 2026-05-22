"""
Benchmark.py — Combined benchmark for GoWish thesis tables.

Runs seven experiments with one command, each with its own flags:

  Experiment 1: Sensitivity sweep (τ/λ grid)                    → Table 8
  Experiment 2: Basic vs. Python Cumulate                        → Table 9
  Experiment 3: Example rules at K=5, s=0.02                     → Table 3
  Experiment 4: K-sweep (K=1..5, all implementations)           → Figures 2,3,4,8 + Tables 5,10
  Experiment 5: Support sweep (s varies, K=3 fixed)              → Figures 5,6 + Table 11
  Experiment 6: L0-to-L0 leaf pair case study                    → Industrial illustration
  Experiment 7: Rule-based candidate-space reduction benchmark   → Figures / reduction stats

Usage:
    python Benchmark.py <merged_parquet_dir> [options]

Example (paper settings, all experiments):
    python Benchmark.py Data/samples/100000 \\
        --output-dir benchmark_results \\
        --catalogue-base Data/gowish_full \\
        --candidate-rules-csv benchmark_results/basic_vs_cumulate/runs/.../rules_python.csv

Skip experiments with --skip:
    python Benchmark.py Data/samples/100000 --skip sensitivity example_rules
"""

import argparse
import ast
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from itertools import combinations as itertools_combinations
from itertools import product as itertools_product
from math import comb as math_comb
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global plot style — compact thesis figures, clean spines
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       160,
})

# Okabe-Ito colorblind-safe palette (used for Figures 4, 6)
OI_BLACK           = "#000000"
OI_ORANGE          = "#E69F00"
OI_SKY_BLUE        = "#56B4E9"
OI_BLUISH_GREEN    = "#009E73"
OI_YELLOW          = "#F0E442"
OI_BLUE            = "#0072B2"
OI_VERMILLION      = "#D55E00"
OI_REDDISH_PURPLE  = "#CC79A7"

# Notebook-exact palette (matches benchmark_thesis_walkthrough.ipynb exactly)
COL_PY     = "#56B4E9"   # Python Cumulate (same as OI_SKY_BLUE)
COL_CPP    = "#009E73"   # C++ Cumulate (same as OI_BLUISH_GREEN)
COL_MLX    = "#D55E00"   # mlxtend flat (same as OI_VERMILLION)
COL_RED    = "#E15759"   # score/hierarchy removed
COL_ORANGE = "#F5A623"   # mirror duplicates removed / mlxtend bars
COL_PURPLE = "#6A5ACD"   # k=5 bar in Fig 2
# Per-bar colors for Figure 2 (incremental rule discovery)
_FIG2_COLORS = ["#4C78A8", "#4CAF50", COL_RED, COL_ORANGE, COL_PURPLE]
# Bar color for Cumulate in Figure 3 and Figure 8
COL_CUMULATE_BAR = "#4C9AE8"

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
    make_level_tokens, parse_token,
)
from frequent_patterns import fpcommon as fpc  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combined benchmark for GoWish thesis (Tables 3, 8, 9 + industrial).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "base",
        help="Directory containing pre-merged parquet(s) "
             "(columns: wishlist_id, category_name).",
    )
    p.add_argument(
        "--mining-base", default=None, metavar="DIR",
        help="Parquet file/directory used for Experiments 1-5. Defaults to <base>. "
             "Use this to run benchmark sweeps on Data/samples/100000 while keeping "
             "--catalogue-base or <base> as the full catalogue for Experiments 6-8.",
    )
    p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Root directory for all benchmark outputs.",
    )
    p.add_argument(
        "--skip", nargs="*",
        choices=[
            "sensitivity", "basic_vs_cumulate", "example_rules",
            "k_sweep", "support_sweep",
            "l0_pair_example", "rule_candidate_space",
            "held_out_recall",
        ],
        default=[],
        help="Experiments to skip. By default all eight run.",
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
        help="Materialize a first-N-wishlists parquet sample for Experiments 1-5. "
             "The sample is written under <output-dir>/_samples and passed to all "
             "subprocess-based benchmark runs.",
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
    g4.add_argument("--ksweep-max-ante-len", type=int, default=3, metavar="A",
                    help="Max antecedent length for k-sweep runs (thesis default: 3).")
    g4.add_argument("--ksweep-max-cons-len", type=int, default=2, metavar="B",
                    help="Max consequent length for k-sweep runs (thesis default: 2).")
    g4.add_argument(
        "--cpp-exe", default=None, metavar="EXE",
        help="Path to compiled apriori_cumulate_cpp binary. "
             "Auto-detected at apriori_cumulate/cpp/apriori_cumulate_cpp when omitted. "
             "Included in k-sweep only when the file exists.",
    )

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
    g5.add_argument("--ssweep-max-ante-len", type=int, default=3, metavar="A",
                    help="Max antecedent length for support-sweep runs (thesis default: 3).")
    g5.add_argument("--ssweep-max-cons-len", type=int, default=2, metavar="B",
                    help="Max consequent length for support-sweep runs (thesis default: 2).")

    # ── Shared catalogue args (experiments 6 & 7) ────────────────────────────
    gc = p.add_argument_group(
        "Catalogue settings — Experiments 6 & 7")
    gc.add_argument(
        "--catalogue-base", default=None, metavar="DIR",
        help="Merged parquet file/directory (wishlist_id, product_id, category_name) "
             "or a directory with products/ and categories/ sub-directories. "
             "Defaults to <base> when omitted.",
    )
    gc.add_argument(
        "--catalogue-k-levels", type=int, default=3, metavar="K",
        help="Hierarchy depth used when tokenising category paths for the catalogue.",
    )

    # ── Experiment 6: L0 pair example ────────────────────────────────────────
    g6 = p.add_argument_group(
        "Experiment 6 — L0-to-L0 leaf pair case study")
    g6.add_argument(
        "--l0pair-antecedent",
        default="Clothing > Shorts > Denim Shorts",
        help="Full category path label for the antecedent leaf.",
    )
    g6.add_argument(
        "--l0pair-consequent",
        default="Activewear > Activewear Tops > T-Shirts",
        help="Full category path label for the consequent leaf.",
    )

    # ── Experiment 7: rule candidate space ───────────────────────────────────
    g7 = p.add_argument_group(
        "Experiment 7 — Rule-based candidate-space reduction")
    g7.add_argument(
        "--candidate-rules-csv", default=None, metavar="CSV",
        help="Path to a Cumulate rule CSV (must have a_toks and b_toks columns). "
             "When omitted, Benchmark.py uses the Experiment 3 example-rules CSV "
             "generated earlier in the same run.",
    )
    g7.add_argument(
        "--candidate-top-examples", type=int, default=20, metavar="N",
        help="Number of high-scoring single-consequent rules to export.",
    )
    g7.add_argument(
        "--candidate-specific-examples", type=int, default=6, metavar="N",
        help="Number of concrete before/after rule examples to plot.",
    )

    g8 = p.add_argument_group("Experiment 8 — Held-out recall evaluation (Figure 12)")
    g8.add_argument(
        "--held-out-train-frac", type=float, default=0.8, metavar="F",
        help="Fraction of wishlists used for training; remainder is the held-out test set.",
    )
    g8.add_argument(
        "--held-out-mine-conf", type=float, default=0.50, metavar="C",
        help="Minimum confidence when mining rules on the training split.",
    )
    g8.add_argument(
        "--held-out-final-conf", type=float, default=0.60, metavar="C",
        help="Highlighted threshold on the recall-reduction plot.",
    )
    g8.add_argument(
        "--held-out-conf-sweep",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        metavar="LIST",
        help="Comma-separated confidence thresholds for the recall-reduction curve.",
    )

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


def _first_ms(text: str, pattern: str) -> Optional[float]:
    """Extract a millisecond timing captured by group 1 of 'pattern' and return seconds."""
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return float(m.group(1)) / 1000.0 if m else None


def _rule_count(metrics: Dict[str, Any]) -> Any:
    final = metrics.get("final_rules")
    if final is not None:
        return final
    raw = metrics.get("raw_rules")
    if raw is not None:
        return raw
    return "?"


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
            text, [r"Raw rules:\s*([0-9,]+)", r"Rules:\s*([0-9,]+)",
                   r"\[6\] Rules \(raw\): ([0-9,]+) in"]),
        "score_rank_rules": _first_int(
            text, [r"After score\+rank:\s*([0-9,]+)\s+rules"]),
        "family_dedupe_rules": _first_int(
            text, [r"After family dedupe:\s*([0-9,]+)\s+rules"]),
        "antimirror_dedupe_rules": _first_int(
            text, [r"After antimirror dedupe:\s*([0-9,]+)\s+rules"]),
        "final_rules": _first_int(
            text, [r"Found\s+([0-9,]+)\s+rules",
                   r"After antimirror dedupe:\s*([0-9,]+)\s+rules"]),
    }
    # C++ binary emits millisecond timings — fill in any gaps not matched above
    cpp_fills: Dict[str, Optional[Any]] = {
        "load_data_seconds":            _first_ms(text, r"\[1\] Loaded \d+ rows in ([0-9.]+) ms"),
        "build_transactions_seconds":   _first_ms(text, r"\[2\] Built \d+ transactions in ([0-9.]+) ms"),
        "encode_transactions_seconds":  _first_ms(text, r"\[3\] Encoded .+ in ([0-9.]+) ms"),
        "build_ancestor_map_seconds":   _first_ms(text, r"\[4b\] Built ancestors for \d+ items in ([0-9.]+) ms"),
        "apriori_seconds":              _first_ms(text, r"\[5\] Apriori: \d+ itemsets in ([0-9.]+) ms"),
        "association_rules_seconds":    _first_ms(text, r"\[6\] Rules \(raw\): \d+ in ([0-9.]+) ms"),
    }
    for _ck, _cv in cpp_fills.items():
        if out.get(_ck) is None and _cv is not None:
            out[_ck] = _cv
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


def _materialize_wishlist_sample(base: str, n_wishlists: int, out_root: Path) -> Path:
    sample_dir = out_root / "_samples" / f"first_{n_wishlists}"
    sample_file = sample_dir / "part-00000.parquet"
    meta_file = sample_dir / "sample_metadata.json"
    if sample_file.exists():
        return sample_dir

    sample_dir.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_data(base))
    keep = set(df["wishlist_id"].drop_duplicates().head(n_wishlists))
    sample = df[df["wishlist_id"].isin(keep)].copy()
    sample.to_parquet(sample_file, index=False)
    meta_file.write_text(
        json.dumps(
            {
                "source": base,
                "method": "first_n_wishlists_after_cleaning",
                "requested_wishlists": int(n_wishlists),
                "actual_wishlists": int(sample["wishlist_id"].nunique()),
                "rows": int(len(sample)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sample_dir


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


def _build_branch_ancestry_from_category_paths(rows: pd.DataFrame) -> Dict[str, Any]:
    taxonomy_paths: set[str] = set()
    for path in rows["category_name"].dropna().unique():
        parts = [p.strip() for p in str(path).split(">") if p.strip()]
        for i in range(1, len(parts) + 1):
            taxonomy_paths.add(" > ".join(parts[:i]))
    return fpc.build_branch_ancestry(
        pd.DataFrame({"category_name": sorted(taxonomy_paths)}),
        name_col="category_name",
    )


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
        df = load_data(args.mining_base)

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
        branch_ancestry = _build_branch_ancestry_from_category_paths(df_min)
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
            args.python_exe, cfg["script"], args.mining_base,
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
            f"rules={_rule_count(metrics)}"
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
            f"{_rule_count(row):>8}"
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
        args.python_exe, str(CUMULATE_PIPELINE), args.mining_base,
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
# Figure helpers — k-sweep and support-sweep plots (Figures 2–9)
# ===========================================================================

def _paper_axes(ax: Any) -> None:
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _make_k_sweep_figures(rows: List[Dict[str, Any]], out_dir: Path, args: Any) -> None:
    """Generate thesis figures from k-sweep row data (Figures 2, 3, 4, 7, 8, 9)."""
    from statistics import median as _median

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    def af(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    good = [r for r in rows if r.get("success")]
    if not good:
        print("  [figures] No successful k-sweep runs — skipping figures.")
        return

    k_values = sorted({int(r["k_levels"]) for r in good})

    def med(field: str, impl: Optional[str] = None, k: Optional[int] = None) -> Optional[float]:
        sub = good
        if impl is not None:
            sub = [r for r in sub if r.get("implementation") == impl]
        if k is not None:
            sub = [r for r in sub if int(r.get("k_levels", -1)) == k]
        vals = [af(r.get(field)) for r in sub]
        vals = [v for v in vals if v is not None]
        return _median(vals) if vals else None

    # ── Figure 2: Incremental rule discovery by k-level ─────────────────────
    cumulate_rules: Dict[int, float] = {}
    for k in k_values:
        v = med("final_rules", impl="python_cumulate", k=k)
        if v is not None:
            cumulate_rules[k] = v

    if len(cumulate_rules) >= 2:
        sorted_k = sorted(cumulate_rules)
        totals = [cumulate_rules[k] for k in sorted_k]
        incremental = [totals[0]] + [totals[i] - totals[i - 1] for i in range(1, len(totals))]
        xpos = list(range(len(sorted_k)))
        bar_colors = [_FIG2_COLORS[i % len(_FIG2_COLORS)] for i in range(len(sorted_k))]
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        bars = ax.bar(xpos, incremental, color=bar_colors, edgecolor="black", linewidth=0.5)
        ymax = max(v for v in incremental if v > 0) if any(v > 0 for v in incremental) else 1
        ax.set_ylim(0, ymax * 1.35)
        for bar, inc, tot in zip(bars, incremental, totals):
            cx = bar.get_x() + bar.get_width() / 2
            ax.text(cx, bar.get_height() + ymax * 0.035,
                    str(int(round(inc))), ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(cx, bar.get_height() + ymax * 0.13,
                    f"total={int(round(tot))}", ha="center", va="bottom", fontsize=7, color="0.55")
        ax.set_title(
            f"Incremental rule discovery by k-level\n"
            f"(support={args.ksweep_support}, total = cumulative)",
            fontweight="bold",
        )
        ax.set_ylabel("New rules added at this k-level")
        ax.set_xticks(xpos, [f"k={k}" for k in sorted_k])
        _paper_axes(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / "thesis_incremental_k.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 3: Final rules remaining by k-level (cumulate vs mlxtend) ────
    mlxtend_rules: Dict[int, float] = {}
    for k in k_values:
        v = med("final_rules", impl="mlxtend_flat", k=k)
        if v is not None:
            mlxtend_rules[k] = v

    if cumulate_rules or mlxtend_rules:
        sorted_k = sorted(set(cumulate_rules) | set(mlxtend_rules))
        xpos = list(range(len(sorted_k)))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        impl_specs = [
            (cumulate_rules, "Cumulate (Python/C++)", COL_CUMULATE_BAR, -width / 2),
            (mlxtend_rules,  "mlxtend (flat)",        COL_ORANGE,        width / 2),
        ]
        y_max = max(
            (max(cumulate_rules.values()) if cumulate_rules else 0),
            (max(mlxtend_rules.values()) if mlxtend_rules else 0),
        )
        for data, label, color, offset in impl_specs:
            ys = [data.get(k, 0) for k in sorted_k]
            bars = ax.bar(
                [x + offset for x in xpos], ys, width=width,
                label=label, color=color, edgecolor="black", linewidth=0.4,
            )
            for bar, y in zip(bars, ys):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.015,
                    str(int(round(y))), ha="center", va="bottom", fontsize=8,
                )
        ax.set_ylim(bottom=0, top=y_max * 1.12)
        ax.set_title(
            f"Final rules remaining by k-level\n"
            f"(support={args.ksweep_support}, median across repeats)",
            fontweight="bold",
        )
        ax.set_ylabel("Final rules (median)")
        ax.set_xticks(xpos, [f"k={k}" for k in sorted_k])
        ax.legend(loc="upper left", frameon=True)
        _paper_axes(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / "final_rules_cumulate_vs_mlxtend_by_k.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 4: Median algorithm runtime by k-level ────────────────────────
    all_impls = [
        ("python_cumulate", "Python Cumulate", OI_SKY_BLUE),
        ("cpp_cumulate",    "C++ Cumulate",    OI_BLUISH_GREEN),
        ("mlxtend_flat",    "mlxtend (flat)",  OI_VERMILLION),
    ]
    present_impls = [
        (impl, lbl, col) for impl, lbl, col in all_impls
        if any(r.get("implementation") == impl for r in good)
    ]
    runtime_by_impl: Dict[str, Dict[int, float]] = {}
    for impl, _lbl, _col in present_impls:
        for k in k_values:
            v = med("algorithm_core_seconds", impl=impl, k=k)
            if v is not None and v > 0:
                runtime_by_impl.setdefault(impl, {})[k] = v

    if runtime_by_impl:
        n = len(present_impls)
        width = min(0.25, 0.80 / n)
        start = -width * (n - 1) / 2
        xpos = list(range(len(k_values)))
        fig, ax = plt.subplots(figsize=(6.2, 3.5))
        for idx, (impl, label, color) in enumerate(present_impls):
            ys = [runtime_by_impl.get(impl, {}).get(k, float("nan")) for k in k_values]
            ax.bar(
                [x + start + idx * width for x in xpos], ys,
                width=width, label=label, color=color, edgecolor="black", linewidth=0.4,
            )
        ax.set_title("Median algorithm runtime by k-level", fontweight="bold")
        ax.set_xlabel("k-level")
        ax.set_ylabel("Algorithm/core time (sec, log scale)")
        ax.set_yscale("log")
        ax.set_xticks(xpos, [str(k) for k in k_values])
        ax.legend(loc="upper left", frameon=True)
        _paper_axes(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / "runtime_by_k_level_python_cpp.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 7: Rule reduction funnel (at K=3 or middle K) ─────────────────
    funnel_k = 3 if 3 in k_values else (k_values[len(k_values) // 2] if k_values else None)
    if funnel_k is not None:
        funnel_fields = [
            ("raw_rules",              "Raw"),
            ("score_rank_rules",       "Lift/score"),
            ("family_dedupe_rules",    "Family"),
            ("antimirror_dedupe_rules","Antimirror"),
            ("final_rules",            "Final"),
        ]
        labels, vals = [], []
        for field, label in funnel_fields:
            v = med(field, impl="python_cumulate", k=funnel_k)
            if v is not None:
                labels.append(label)
                vals.append(v)
        if len(vals) >= 2:
            fig, ax = plt.subplots(figsize=(6.2, 2.8))
            ax.plot(range(len(vals)), vals, color="#7B68EE",
                    marker="o", linewidth=1.1, markersize=3)
            ax.set_title(
                f"Rule reduction through post-processing\n"
                f"k={funnel_k}, support={args.ksweep_support:.2f}, "
                f"confidence={args.ksweep_conf:.2f}, lift={args.ksweep_lift:.2f}",
                fontweight="bold",
            )
            ax.set_xlabel("Stage")
            ax.set_ylabel("Rules")
            ax.set_xticks(range(len(vals)), labels)
            _paper_axes(ax)
            fig.tight_layout()
            fig.savefig(figs_dir / "rule_reduction_funnel.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    # ── Figure 8: Stacked redundancy breakdown by k ───────────────────────────
    stack: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        final  = med("final_rules",            impl="python_cumulate", k=k)
        family = med("family_dedupe_rules",     impl="python_cumulate", k=k)
        score  = med("score_rank_rules",        impl="python_cumulate", k=k)
        raw    = med("raw_rules",               impl="python_cumulate", k=k)
        if final is not None:
            mirror   = max(0.0, (family - final)  if family is not None else 0.0)
            sc_hier  = max(0.0, (raw    - score)   if raw is not None and score is not None else 0.0)
            stack[k] = {"final": final, "mirror": mirror, "sc_hier": sc_hier}

    if stack:
        sorted_k = sorted(stack)
        xpos = list(range(len(sorted_k)))
        fig, ax = plt.subplots(figsize=(6.2, 3.7))
        finals  = [stack[k]["final"]   for k in sorted_k]
        mirrors = [stack[k]["mirror"]  for k in sorted_k]
        scores  = [stack[k]["sc_hier"] for k in sorted_k]
        ax.bar(xpos, finals,  color=COL_CUMULATE_BAR, edgecolor="black", linewidth=0.4,
               label="Final rules (kept)")
        ax.bar(xpos, mirrors, bottom=finals, color=COL_ORANGE, edgecolor="black", linewidth=0.4,
               label="Mirror duplicates removed")
        ax.bar(xpos, scores,  bottom=[f + m for f, m in zip(finals, mirrors)],
               color=COL_RED, edgecolor="black", linewidth=0.4,
               label="Score/hierarchy redundancy removed")
        for x, f in zip(xpos, finals):
            if f > 100:
                ax.text(x, f / 2, str(int(round(f))), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        ax.set_title(
            f"Cumulate rule filtering by k-level\n(support={args.ksweep_support})",
            fontweight="bold",
        )
        ax.set_ylabel("Rule count")
        ax.set_xticks(xpos, [f"k={k}" for k in sorted_k])
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        _paper_axes(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / "thesis_redundancy_stacked.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 9: Phase breakdown — all k-levels × all implementations ────────
    phase_specs = [
        ("apriori_seconds",             "apriori"),
        ("postprocess_seconds",         "post"),
        ("encode_transactions_seconds", "encoding"),
        ("association_rules_seconds",   "rules"),
        ("build_transactions_seconds",  "transactions"),
        ("build_ancestor_map_seconds",  "ancestor"),
    ]
    # Phase colors exactly matching benchmark_thesis_walkthrough.ipynb Figure 9
    phase_colors = [COL_RED, "#8C564B", COL_ORANGE, "#756BB1", COL_PY, "#4CAF50"]
    # Bar order matches thesis: cpp first, then mlx, then py
    impl_short = [
        ("cpp_cumulate",    "cpp"),
        ("mlxtend_flat",    "mlx"),
        ("python_cumulate", "py"),
    ]
    bar_labels: List[str] = []
    phase_vals: Dict[str, List[float]] = {name: [] for _, name in phase_specs}
    for k in sorted(k_values):
        for impl, short in impl_short:
            if not any(r.get("implementation") == impl for r in good):
                continue
            bar_labels.append(f"k{k}\n{short}")
            for field, name in phase_specs:
                phase_vals[name].append(med(field, impl=impl, k=k) or 0.0)
    if bar_labels:
        fig, ax = plt.subplots(figsize=(6.2, 3.52))
        bottom = np.zeros(len(bar_labels))
        for (field, name), color in zip(phase_specs, phase_colors):
            vals_p = np.array(phase_vals[name])
            ax.bar(bar_labels, vals_p, bottom=bottom, label=name, color=color,
                   edgecolor="black", linewidth=0.2)
            bottom += vals_p
        ax.set_ylim(0, bottom.max() * 1.15)
        ax.set_title(
            f"Phase breakdown (support={args.ksweep_support}, confidence={args.ksweep_conf})",
            fontweight="bold",
        )
        ax.set_ylabel("seconds")
        ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1), frameon=True, fontsize=7)
        _paper_axes(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / "phase_breakdown_baseline.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"  [figures] Saved k-sweep figures to {figs_dir}")


def _make_support_sweep_figures(
    rows: List[Dict[str, Any]], out_dir: Path, args: Any, k_sweep_csv: Optional[Path] = None
) -> None:
    """Generate thesis figures from support-sweep data (Figures 5, 6)."""
    from statistics import median as _median

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    def af(v: Any) -> Optional[float]:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    good = [r for r in rows if r.get("success")]
    if not good:
        print("  [figures] No successful support-sweep runs — skipping figures.")
        return

    supports = sorted({float(r["min_support"]) for r in good}, reverse=True)

    def med_s(field: str, impl: str = "python_cumulate", s: Optional[float] = None) -> Optional[float]:
        sub = [r for r in good if r.get("implementation") == impl]
        if s is not None:
            sub = [r for r in sub if float(r.get("min_support", -1)) == s]
        vals = [af(r.get(field)) for r in sub]
        vals = [v for v in vals if v is not None]
        return _median(vals) if vals else None

    def fmt_s(s: float) -> str:
        return f"{s:.2f}".rstrip("0").rstrip(".")

    # ── Figure 5: Python minimum support sensitivity ──────────────────────────
    xpos = list(range(len(supports)))
    final_r  = [med_s("final_rules",               s=s) for s in supports]
    raw_r    = [med_s("raw_rules",                  s=s) for s in supports]
    assoc_t  = [med_s("association_rules_seconds",  s=s) for s in supports]

    if any(v is not None for v in final_r):
        fig, ax1 = plt.subplots(figsize=(6.2, 3.5))
        for ys, label, marker, ls, color in [
            (final_r, "Final rules", "o", "-",  "#3A8DFF"),
            (raw_r,   "Raw rules",   "s", "--", COL_RED),
        ]:
            if any(v is not None for v in ys):
                ax1.plot(xpos, [float("nan") if v is None else v for v in ys],
                         color=color, marker=marker, linestyle=ls, label=label, linewidth=1.5,
                         markersize=4)
        ax1.set_xlabel("Minimum support")
        ax1.set_ylabel("Rules")
        ax1.set_xticks(xpos, [fmt_s(s) for s in supports])
        _paper_axes(ax1)

        ax2 = ax1.twinx()
        if any(v is not None for v in assoc_t):
            ax2.plot(xpos, [float("nan") if v is None else v for v in assoc_t],
                     color=COL_CPP, marker="^", linestyle=":", label="Rule generation time",
                     linewidth=1.2, markersize=4)
        ax2.set_ylabel("Rule generation time (sec)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc="upper left", fontsize=8)
        ax1.set_title("Python minimum support sensitivity", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figs_dir / "python_support_sensitivity.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 6: Efficiency scatter (rules found vs algorithm time) ──────────
    # Combine support-sweep rows (all implementations) with k-sweep rows.
    # Thesis caption: "all three implementations across K∈{1..5} and s∈{0.05..0.01}"
    all_rows: List[Dict[str, Any]] = list(good)
    if k_sweep_csv is not None and k_sweep_csv.exists():
        try:
            all_rows.extend(pd.read_csv(k_sweep_csv).to_dict(orient="records"))
        except Exception:
            pass

    # Aggregate repeats: median time per unique (implementation, k_levels, min_support) condition
    if all_rows:
        _df = pd.DataFrame(all_rows)
        _grp = [c for c in ["implementation", "k_levels", "min_support"] if c in _df.columns]
        _agg: Dict[str, Any] = {"algorithm_core_seconds": "median"}
        if "final_rules" in _df.columns:
            _agg["final_rules"] = "first"
        all_rows = _df.groupby(_grp, as_index=False).agg(_agg).to_dict(orient="records")

    impl_styles: Dict[str, Dict[str, Any]] = {
        "cpp_cumulate":    {"color": OI_BLUISH_GREEN, "marker": "o", "label": "C++ Cumulate"},
        "python_cumulate": {"color": OI_SKY_BLUE,     "marker": "^", "label": "Python Cumulate"},
        "mlxtend_flat":    {"color": OI_VERMILLION,   "marker": "s", "label": "mlxtend (flat)"},
    }
    scatter: Dict[str, tuple] = {}
    for r in all_rows:
        impl = str(r.get("implementation", ""))
        if impl not in impl_styles:
            continue
        x = af(r.get("algorithm_core_seconds"))
        y = af(r.get("final_rules"))
        if x and x > 0 and y and y > 0:
            if impl not in scatter:
                scatter[impl] = ([], [])
            scatter[impl][0].append(x)
            scatter[impl][1].append(y)

    if scatter:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        # Plot in thesis legend order: C++, Python, mlxtend
        for impl in ["cpp_cumulate", "python_cumulate", "mlxtend_flat"]:
            if impl not in scatter:
                continue
            xs, ys = scatter[impl]
            style = impl_styles[impl]
            marker_size = 58 if style["marker"] == "^" else 52
            ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                       label=style["label"], s=marker_size,
                       edgecolor="black", linewidth=0.45, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Algorithm/core time (s, log scale)")
        ax.set_ylabel("Final rules found (log scale)")
        ax.set_title(
            f"Efficiency: rules found vs algorithm time\n(K-sweep, support={args.ksweep_support})",
            fontweight="bold",
        )
        ax.legend(loc="upper left", frameon=True)
        ax.grid(False)
        _paper_axes(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / "thesis_efficiency_scatter.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"  [figures] Saved support-sweep figures to {figs_dir}")


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

    _cpp_exe_path = Path(getattr(args, "cpp_exe", None) or
                         REPO_ROOT / "apriori_cumulate" / "cpp" / "apriori_cumulate_cpp")
    _ksweep_max_len = args.ksweep_max_ante_len + args.ksweep_max_cons_len
    implementations = [
        ("python_cumulate", str(CUMULATE_PIPELINE), [
            "--max-ante-len", str(args.ksweep_max_ante_len),
            "--max-cons-len", str(args.ksweep_max_cons_len),
            "--max-len",      str(_ksweep_max_len),
        ]),
        ("mlxtend_flat", str(MLX_FLAT_PIPELINE), [
            "--max-len", str(_ksweep_max_len),
        ]),
    ]
    if _cpp_exe_path.exists():
        implementations.append(("cpp_cumulate", str(_cpp_exe_path), []))

    # ── Warmup pass ──────────────────────────────────────────────────────────
    # Run each implementation once before timing begins so that OS page caches
    # are warm and shared libraries (Arrow, Parquet, numpy) are resident in RAM.
    # Results are discarded; only the subsequent timed runs are recorded.
    # This is standard practice in systems benchmarking — see paper §methodology.
    warmup_k   = args.ksweep_k[len(args.ksweep_k) // 2]   # middle K value
    warmup_dir = out_dir / "_warmup"
    print(f"  [warmup] Untimed pass per implementation at K={warmup_k} "
          f"(warms OS cache + shared libs) …")
    for impl, script, extra_args in implementations:
        wdir = warmup_dir / impl
        wdir.mkdir(parents=True, exist_ok=True)
        if impl == "mlxtend_flat":
            wcmd = [
                args.python_exe, script, args.mining_base,
                "--min-support", str(args.ksweep_support),
                "--min-conf",    str(args.ksweep_conf),
                "--min-lift",    str(args.ksweep_lift),
                "--output",      str(wdir / "rules.csv"),
            ] + extra_args
        elif impl == "cpp_cumulate":
            wcmd = [
                script, args.mining_base,
                str(warmup_k),
                str(args.ksweep_support),
                str(args.ksweep_conf),
                str(args.ksweep_lift),
                str(args.ksweep_max_ante_len),
                str(args.ksweep_max_cons_len),
                str(wdir / "rules.csv"),
            ]
        else:
            wcmd = [
                args.python_exe, script, args.mining_base,
                "--k-levels",    str(warmup_k),
                "--min-support", str(args.ksweep_support),
                "--min-conf",    str(args.ksweep_conf),
                "--min-lift",    str(args.ksweep_lift),
                "--output",      str(wdir / "rules.csv"),
            ] + extra_args
        run_subprocess(wcmd, wdir)   # result discarded
    print("  [warmup] Done. Starting timed runs.\n")

    rows: List[Dict[str, Any]] = []
    for k in args.ksweep_k:
        for repeat_id in range(1, args.repeats + 1):
            for impl, script, extra_args in implementations:
                run_id    = f"{impl}_k{k}_r{repeat_id}"
                run_dir   = out_dir / "runs" / run_id
                rules_file = run_dir / "rules.csv"

                # mlxtend_flat has no --k-levels flag; cpp_cumulate takes positional args
                if impl == "mlxtend_flat":
                    command = [
                        args.python_exe, script, args.mining_base,
                        "--min-support", str(args.ksweep_support),
                        "--min-conf",    str(args.ksweep_conf),
                        "--min-lift",    str(args.ksweep_lift),
                        "--output",      str(rules_file),
                    ] + extra_args
                elif impl == "cpp_cumulate":
                    # C++ binary: <exe> <base> <k> <support> <conf> <lift> <max_ante> <max_cons>
                    command = [
                        script, args.mining_base,
                        str(k),
                        str(args.ksweep_support),
                        str(args.ksweep_conf),
                        str(args.ksweep_lift),
                        str(args.ksweep_max_ante_len),
                        str(args.ksweep_max_cons_len),
                        str(rules_file),
                    ]
                else:
                    command = [
                        args.python_exe, script, args.mining_base,
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
                    f"rules={_rule_count(metrics)}"
                )
                if not metrics["success"]:
                    print(f"    stderr → {run_dir / 'stderr.log'}")
                rows.append(metrics)

    _save_csv(rows, out_dir / "k_sweep_summary.csv")
    _print_sweep_table(rows, group_col="k_levels", group_label="K")
    _make_k_sweep_figures(rows, out_dir, args)


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

    # ── Warmup pass ──────────────────────────────────────────────────────────
    # Untimed run to warm OS page cache and Python/Arrow shared libraries
    # before any timed support-sweep measurements begin.
    _warmup_max_len = args.ssweep_max_ante_len + args.ssweep_max_cons_len
    warmup_dir = out_dir / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [warmup] Untimed pass per implementation (K={args.ssweep_k}, s={args.ssweep_support[0]}) …")
    _w_cpp = Path(getattr(args, "cpp_exe", None) or
                  REPO_ROOT / "apriori_cumulate" / "cpp" / "apriori_cumulate_cpp")
    _w_s = args.ssweep_support[0]
    for _wcmd in [
        [args.python_exe, str(CUMULATE_PIPELINE), args.mining_base,
         "--k-levels", str(args.ssweep_k), "--min-support", str(_w_s),
         "--min-conf", str(args.ssweep_conf), "--min-lift", str(args.ssweep_lift),
         "--max-ante-len", str(args.ssweep_max_ante_len),
         "--max-cons-len", str(args.ssweep_max_cons_len),
         "--max-len", str(_warmup_max_len),
         "--output", str(warmup_dir / "rules_py.csv")],
        [args.python_exe, str(MLX_FLAT_PIPELINE), args.mining_base,
         "--min-support", str(_w_s), "--min-conf", str(args.ssweep_conf),
         "--min-lift", str(args.ssweep_lift),
         "--max-len", str(_warmup_max_len),
         "--output", str(warmup_dir / "rules_mlx.csv")],
    ] + ([
        [str(_w_cpp), args.mining_base, str(args.ssweep_k), str(_w_s),
         str(args.ssweep_conf), str(args.ssweep_lift),
         str(args.ssweep_max_ante_len), str(args.ssweep_max_cons_len),
         str(warmup_dir / "rules_cpp.csv")]
    ] if _w_cpp.exists() else []):
        run_subprocess(_wcmd, warmup_dir)   # result discarded
    print("  [warmup] Done. Starting timed runs.\n")

    _ssweep_max_len = args.ssweep_max_ante_len + args.ssweep_max_cons_len
    _cpp_exe_path = Path(getattr(args, "cpp_exe", None) or
                         REPO_ROOT / "apriori_cumulate" / "cpp" / "apriori_cumulate_cpp")
    # All three implementations — matches thesis Fig 6 which plots all across s sweep
    ss_implementations = [
        ("python_cumulate", str(CUMULATE_PIPELINE), [
            "--max-ante-len", str(args.ssweep_max_ante_len),
            "--max-cons-len", str(args.ssweep_max_cons_len),
            "--max-len",      str(_ssweep_max_len),
        ]),
        ("mlxtend_flat", str(MLX_FLAT_PIPELINE), [
            "--max-len", str(_ssweep_max_len),
        ]),
    ]
    if _cpp_exe_path.exists():
        ss_implementations.append(("cpp_cumulate", str(_cpp_exe_path), []))

    rows: List[Dict[str, Any]] = []
    for s in args.ssweep_support:
        for repeat_id in range(1, args.repeats + 1):
            for impl, script, extra_args in ss_implementations:
                s_str      = str(s).replace(".", "p")
                run_id     = f"{impl}_s{s_str}_r{repeat_id}"
                run_dir    = out_dir / "runs" / run_id
                rules_file = run_dir / "rules.csv"

                if impl == "mlxtend_flat":
                    command = [
                        args.python_exe, script, args.mining_base,
                        "--min-support", str(s),
                        "--min-conf",    str(args.ssweep_conf),
                        "--min-lift",    str(args.ssweep_lift),
                        "--output",      str(rules_file),
                    ] + extra_args
                elif impl == "cpp_cumulate":
                    command = [
                        script, args.mining_base,
                        str(args.ssweep_k),
                        str(s),
                        str(args.ssweep_conf),
                        str(args.ssweep_lift),
                        str(args.ssweep_max_ante_len),
                        str(args.ssweep_max_cons_len),
                        str(rules_file),
                    ]
                else:
                    command = [
                        args.python_exe, script, args.mining_base,
                        "--k-levels",     str(args.ssweep_k),
                        "--min-support",  str(s),
                        "--min-conf",     str(args.ssweep_conf),
                        "--min-lift",     str(args.ssweep_lift),
                        "--output",       str(rules_file),
                    ] + extra_args

                print(f"  {impl}  s={s}  repeat={repeat_id} …", end=" ", flush=True)
                metrics = run_subprocess(command, run_dir)
                metrics.update({
                    "implementation": impl,
                    "k_levels":       args.ssweep_k,
                    "min_support":    s,
                    "repeat_id":      repeat_id,
                    "experiment":     "support_sweep",
                })

                status = "OK" if metrics["success"] else f"FAILED (rc={metrics['return_code']})"
                print(
                    f"{status}  wall={metrics['runtime_seconds']:.1f}s  "
                    f"apriori={metrics.get('apriori_seconds') or 0:.3f}s  "
                    f"rules={_rule_count(metrics)}"
                )
                if not metrics["success"]:
                    print(f"    stderr → {run_dir / 'stderr.log'}")
                rows.append(metrics)

    _save_csv(rows, out_dir / "support_sweep_summary.csv")
    _print_sweep_table(rows, group_col="min_support", group_label="support")
    k_sweep_csv = out_dir.parent / "k_sweep" / "k_sweep_summary.csv"
    _make_support_sweep_figures(rows, out_dir, args, k_sweep_csv=k_sweep_csv)


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
            f"{_rule_count(row):>8}"
        )


# ===========================================================================
# Shared catalogue helpers (experiments 6 & 7)
# ===========================================================================

def _parquet_parts(folder: Path) -> List[Path]:
    return sorted(
        folder / name
        for name in os.listdir(folder)
        if name.endswith(".parquet") and not name.startswith("._")
    )


def _read_parts(folder: Path, columns: List[str]) -> pd.DataFrame:
    parts = _parquet_parts(folder)
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in parts],
        ignore_index=True,
    )


def _load_catalogue(base: Path) -> pd.DataFrame:
    if (base / "products").is_dir() and (base / "categories").is_dir():
        products = _read_parts(
            base / "products",
            ["product_id", "mongo_product_id", "title", "description"],
        )
        products = products[
            products["mongo_product_id"].notna()
            & (products["title"].notna() | products["description"].notna())
        ][["product_id", "mongo_product_id"]]

        categories = _read_parts(base / "categories", ["id", "category_name"])
        categories = categories[categories["category_name"].notna()][["id", "category_name"]]

        catalogue = (
            products.merge(
                categories.rename(columns={"id": "mongo_product_id"}),
                on="mongo_product_id",
                how="inner",
            )[["product_id", "category_name"]]
            .drop_duplicates()
        )
    else:
        # Merged parquet format: wishlist_id, product_id, category_name
        if base.is_dir():
            parts = _parquet_parts(base)
            if not parts:
                raise FileNotFoundError(f"No parquet files found in {base}")
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        else:
            df = pd.read_parquet(base)
        catalogue = df[["product_id", "category_name"]].drop_duplicates()

    if catalogue.empty:
        raise RuntimeError("No categorized products found.")
    return catalogue


def _build_token_products(catalogue: pd.DataFrame, k_levels: int) -> pd.DataFrame:
    token_products = catalogue.copy()
    token_products["token"] = token_products["category_name"].apply(
        lambda path: make_level_tokens(path, k_levels)
    )
    token_products = token_products.explode("token").dropna(subset=["token"])
    return token_products[["product_id", "token"]].drop_duplicates()


def _build_category_inventory(token_products: pd.DataFrame, universe_size: int) -> pd.DataFrame:
    inventory = (
        token_products.groupby("token", as_index=False)["product_id"]
        .nunique()
        .rename(columns={"product_id": "unique_products"})
    )
    inventory[["level", "branch", "label"]] = inventory["token"].apply(
        lambda tok: pd.Series(parse_token(tok))
    )
    inventory["catalogue_share"] = inventory["unique_products"] / universe_size
    return inventory[
        ["token", "label", "level", "branch", "unique_products", "catalogue_share"]
    ].sort_values(["unique_products", "label"], ascending=[False, True])


# ===========================================================================
# EXPERIMENT 6 — L0-to-L0 leaf pair case study
# ===========================================================================

def _find_l0_token(vocab: List[str], label: str) -> str:
    matches = [tok for tok in vocab if tok.startswith("L0|") and label_of(tok) == label]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one L0 token for {label!r}, found {len(matches)}"
        )
    return matches[0]


def _save_l0_pair_chart(row: pd.Series, output_path: Path) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 3.8),
        gridspec_kw={"width_ratios": [1.05, 1.45]},
    )
    ax0, ax1 = axes
    before    = int(row["catalogue_products_before_rule"])
    after     = int(row["consequent_candidate_products"])
    reduction = float(row["search_space_reduction_pct"])

    ant_short  = str(row["antecedent_label"]).split(" > ")[-1]
    cons_label = str(row["consequent_label"])
    cons_short = (
        "Activewear T-Shirts"
        if cons_label.endswith("Activewear Tops > T-Shirts")
        else cons_label.split(" > ")[-1]
    )
    fig.suptitle(f"Leaf-level pair case study: {ant_short} -> {cons_short}", y=0.98)

    metric_labels = ["Support", "Confidence", "Lift"]
    metric_values = [
        f"{100 * float(row['support']):.1f}%",
        f"{100 * float(row['confidence']):.1f}%",
        f"{float(row['lift']):.2f}",
    ]
    ax0.axis("off")
    for idx, (lbl, val) in enumerate(zip(metric_labels, metric_values)):
        y = 0.78 - idx * 0.24
        ax0.text(0.0, y,        lbl, fontsize=10,  color="#59636e")
        ax0.text(0.0, y - 0.10, val, fontsize=18, fontweight="bold", color="#1f2933")

    ax1.barh([""], [after], color="#2f6f6d")
    ax1.set_xlim(0, before * 0.06)
    ax1.set_yticks([])
    ax1.set_xlabel("Remaining candidate products")
    ax1.set_title(f"{after:,} of {before:,} products remain")
    ax1.text(after * 1.03, 0, f"{reduction:.1f}% fewer", va="center", fontsize=11)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.grid(axis="x", color="#d9dee3", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_l0_industrial_overview(
    row: pd.Series,
    inventory: pd.DataFrame,
    output_path: Path,
) -> None:
    before      = int(row["catalogue_products_before_rule"])
    after       = int(row["consequent_candidate_products"])
    reduction   = float(row["search_space_reduction_pct"])
    ant_label   = str(row["antecedent_label"])
    cons_label  = str(row["consequent_label"])
    ant_short   = ant_label.split(" > ")[-1]
    cons_short  = (
        "Activewear T-Shirts"
        if cons_label.endswith("Activewear Tops > T-Shirts")
        else cons_label.split(" > ")[-1]
    )

    ant_products  = int(inventory.loc[inventory["label"].eq(ant_label),  "unique_products"].iloc[0])
    cons_products = int(inventory.loc[inventory["label"].eq(cons_label), "unique_products"].iloc[0])

    fig, axes = plt.subplots(
        1, 3, figsize=(14.6, 4.8),
        gridspec_kw={"width_ratios": [1.18, 0.82, 1.0]},
    )
    ax0, ax1, ax2 = axes
    fig.suptitle("From category pair to smaller recommendation candidate set", y=0.98)

    minimum = max(1, int(inventory["unique_products"].min()))
    maximum = int(inventory["unique_products"].max())
    bins = np.logspace(np.log10(minimum), np.log10(maximum), 30)
    ax0.hist(inventory["unique_products"], bins=bins, color="#d9e1e4", edgecolor="white")
    ax0.set_xscale("log")
    ax0.axvline(ant_products,  color="#c46d3b", linewidth=2)
    ax0.axvline(cons_products, color="#2f6f6d", linewidth=2)
    ax0.text(
        ant_products, ax0.get_ylim()[1] * 0.92,
        f"{ant_short}\n{ant_products:,}",
        ha="center", va="top", fontsize=9, color="#8a4a27",
    )
    ax0.text(
        cons_products, ax0.get_ylim()[1] * 0.62,
        f"{cons_short}\n{cons_products:,}",
        ha="center", va="top", fontsize=9, color="#245452",
    )
    ax0.set_title("1. Category inventory")
    ax0.set_xlabel("Unique products per category token (log scale)")
    ax0.set_ylabel("Number of category tokens")
    ax0.spines[["top", "right"]].set_visible(False)

    ax1.axis("off")
    ax1.set_title("2. Pair signal")
    ax1.text(
        0.5, 0.82, f"{ant_short}\n->\n{cons_short}",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1f2933",
    )
    metric_rows = [
        ("Co-occurring wishlists", f"{int(row['cooccurring_wishlist_count']):,}"),
        ("Support",    f"{100 * float(row['support']):.1f}%"),
        ("Confidence", f"{100 * float(row['confidence']):.1f}%"),
        ("Lift",       f"{float(row['lift']):.2f}"),
    ]
    for idx, (lbl, val) in enumerate(metric_rows):
        y = 0.45 - idx * 0.12
        ax1.text(0.02, y, lbl, fontsize=9.5, color="#59636e")
        ax1.text(0.98, y, val, ha="right", fontsize=10.5, color="#1f2933")

    ax2.barh(["Full catalogue", cons_short], [before, after], color=["#d9e1e4", "#2f6f6d"])
    ax2.set_xlim(0, before * 1.08)
    ax2.set_title("3. Candidate filter")
    ax2.set_xlabel("Candidate products")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.grid(axis="x", color="#d9dee3", linewidth=0.8)
    ax2.tick_params(axis="y", length=0)
    ax2.text(before * 0.99, 0, f"{before:,}", ha="right", va="center", fontsize=10)
    ax2.text(after + before * 0.025, 1, f"{after:,}", va="center", fontsize=10)
    ax2.text(
        0, 1.45, f"{reduction:.1f}% fewer products to score",
        fontsize=12, fontweight="bold", color="#1f2933",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_l0_pair_example(args: argparse.Namespace, out_dir: Path) -> None:
    catalogue_base = Path(args.catalogue_base or args.base)
    k = args.catalogue_k_levels

    print(f"\n{'='*60}")
    print("EXPERIMENT 6: L0-to-L0 leaf pair case study")
    print(f"  antecedent : {args.l0pair_antecedent}")
    print(f"  consequent : {args.l0pair_consequent}")
    print(f"  K          : {k}")
    print(f"  catalogue  : {catalogue_base}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    with timed_step("Loading wishlist data"):
        df = load_data(args.base)
    with timed_step("Cleaning data"):
        df = clean_data(df)

    if args.sample:
        wishlists = df["wishlist_id"].unique()[:args.sample]
        df = df[df["wishlist_id"].isin(wishlists)]

    with timed_step("Building transactions"):
        txns, vocab = build_transactions(df, k)

    sets             = [set(txn) for txn in txns]
    antecedent_tok   = _find_l0_token(vocab, args.l0pair_antecedent)
    consequent_tok   = _find_l0_token(vocab, args.l0pair_consequent)

    n          = len(sets)
    ant_count  = sum(antecedent_tok in txn for txn in sets)
    cons_count = sum(consequent_tok in txn for txn in sets)
    both_count = sum(antecedent_tok in txn and consequent_tok in txn for txn in sets)
    support    = both_count / n
    confidence = both_count / ant_count
    lift       = confidence / (cons_count / n)

    with timed_step("Loading catalogue"):
        catalogue     = _load_catalogue(catalogue_base)
        universe_size = int(catalogue["product_id"].nunique())
        token_products = _build_token_products(catalogue, k)
        inventory      = _build_category_inventory(token_products, universe_size)

    l0_inventory = inventory[inventory["token"].str.startswith("L0|", na=False)]
    candidate_count = int(
        l0_inventory.loc[l0_inventory["token"].eq(consequent_tok), "unique_products"].iloc[0]
    )
    reduction = 100.0 * (1.0 - candidate_count / universe_size)

    row = pd.Series({
        "antecedent_label":               args.l0pair_antecedent,
        "consequent_label":               args.l0pair_consequent,
        "transactions":                   n,
        "antecedent_wishlist_count":      ant_count,
        "consequent_wishlist_count":      cons_count,
        "cooccurring_wishlist_count":     both_count,
        "support":                        support,
        "confidence":                     confidence,
        "lift":                           lift,
        "catalogue_products_before_rule": universe_size,
        "consequent_candidate_products":  candidate_count,
        "products_removed_from_search":   universe_size - candidate_count,
        "search_space_reduction_pct":     reduction,
    })
    row.to_frame().T.to_csv(out_dir / "l0_pair_example.csv", index=False)
    _save_l0_pair_chart(row, out_dir / "l0_pair_example.png")
    _save_l0_industrial_overview(row, l0_inventory, out_dir / "industrial_pair_overview.png")

    print(f"{args.l0pair_antecedent} -> {args.l0pair_consequent}")
    print(f"Support: {support:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Lift: {lift:.4f}")
    print(f"Candidate products: {candidate_count:,} of {universe_size:,}")
    print(f"Search-space reduction: {reduction:.1f}%")
    print(f"Saved outputs to {out_dir}/")


# ===========================================================================
# EXPERIMENT 7 — Rule-based candidate-space reduction
# ===========================================================================

def _parse_token_tuple(value: object) -> tuple:
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, str):
        return (parsed,)
    return tuple(parsed)


def _build_rule_benchmark(
    rules: pd.DataFrame,
    token_products: pd.DataFrame,
    universe_size: int,
) -> pd.DataFrame:
    required = {"rule", "a_toks", "b_toks"}
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"Rule file is missing required columns: {sorted(missing)}")

    token_to_products = (
        token_products.groupby("token")["product_id"]
        .apply(lambda s: frozenset(s))
        .to_dict()
    )

    missing_consequent_tokens: set[str] = set()
    missing_antecedent_tokens: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for row in rules.itertuples(index=False):
        antecedents = _parse_token_tuple(getattr(row, "a_toks"))
        consequents = _parse_token_tuple(getattr(row, "b_toks"))
        missing_antecedent_tokens.update(tok for tok in antecedents if tok not in token_to_products)
        missing_consequent_tokens.update(tok for tok in consequents if tok not in token_to_products)

        antecedent_products = set().union(
            *(token_to_products.get(tok, frozenset()) for tok in antecedents)
        )
        consequent_products = set().union(
            *(token_to_products.get(tok, frozenset()) for tok in consequents)
        )

        candidate_count = len(consequent_products)
        candidate_share = candidate_count / universe_size
        rows.append({
            "rule":                                  getattr(row, "rule"),
            "antecedent_tokens":                     " + ".join(antecedents),
            "consequent_tokens":                     " + ".join(consequents),
            "antecedent_labels":                     " + ".join(label_of(tok) for tok in antecedents),
            "consequent_labels":                     " + ".join(label_of(tok) for tok in consequents),
            "n_antecedent_tokens":                   len(antecedents),
            "n_consequent_tokens":                   len(consequents),
            "antecedent_catalogue_products":         len(antecedent_products),
            "consequent_candidate_products":         candidate_count,
            "candidate_share_of_categorized_catalogue": candidate_share,
            "search_space_reduction_pct":            100.0 * (1.0 - candidate_share),
            "support":                               getattr(row, "support", None),
            "confidence":                            getattr(row, "confidence", None),
            "lift":                                  getattr(row, "lift", None),
            "score":                                 getattr(row, "score", None),
        })

    if missing_consequent_tokens:
        examples = ", ".join(sorted(missing_consequent_tokens)[:5])
        raise ValueError(
            "Rule consequents contain tokens that are not present in the catalogue token inventory. "
            "This usually means --catalogue-k-levels does not match the K used to mine the rules. "
            f"Missing consequent tokens: {examples}"
        )
    if missing_antecedent_tokens:
        examples = ", ".join(sorted(missing_antecedent_tokens)[:5])
        print(
            "[warning] Some rule antecedent tokens were not present in the catalogue token inventory: "
            f"{examples}"
        )

    benchmark = pd.DataFrame(rows)
    return benchmark.sort_values(
        ["search_space_reduction_pct", "score", "confidence", "support"],
        ascending=[False, False, False, False],
    )


def _build_candidate_summary(
    catalogue: pd.DataFrame,
    inventory: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    single = benchmark[benchmark["n_consequent_tokens"] == 1]
    return pd.DataFrame([{
        "categorized_products":                            int(catalogue["product_id"].nunique()),
        "category_tokens":                                 int(len(inventory)),
        "rules":                                           int(len(benchmark)),
        "median_unique_products_per_category":             float(inventory["unique_products"].median()),
        "median_consequent_candidate_products":            float(benchmark["consequent_candidate_products"].median()),
        "median_candidate_share_pct":                      float(100.0 * benchmark["candidate_share_of_categorized_catalogue"].median()),
        "median_search_space_reduction_pct":               float(benchmark["search_space_reduction_pct"].median()),
        "p25_search_space_reduction_pct":                  float(benchmark["search_space_reduction_pct"].quantile(0.25)),
        "p75_search_space_reduction_pct":                  float(benchmark["search_space_reduction_pct"].quantile(0.75)),
        "single_consequent_rules":                         int(len(single)),
        "single_consequent_median_candidate_products":     float(single["consequent_candidate_products"].median()),
        "single_consequent_median_candidate_share_pct":    float(100.0 * single["candidate_share_of_categorized_catalogue"].median()),
        "single_consequent_median_search_space_reduction_pct": float(single["search_space_reduction_pct"].median()),
    }])


def _save_category_histogram(inventory: pd.DataFrame, output_path: Path) -> None:
    minimum = max(1, int(inventory["unique_products"].min()))
    maximum = int(inventory["unique_products"].max())
    bins = np.logspace(np.log10(minimum), np.log10(maximum), 35)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(inventory["unique_products"], bins=bins, color="#2f6f6d", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Unique catalogue products in category token (log scale)")
    ax.set_ylabel("Number of category tokens")
    ax.set_title("Catalogue product counts across hierarchy-aware categories")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_rule_reduction_curve(
    benchmark: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    values = np.sort(benchmark["search_space_reduction_pct"].to_numpy())
    share  = 100.0 * np.arange(1, len(values) + 1) / len(values)
    median = float(np.median(values))
    p10    = float(np.quantile(values, 0.10))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(values, share, color="#2f6f6d", linewidth=2.5)
    ax.fill_between(values, share, color="#2f6f6d", alpha=0.12)
    ax.axvline(median, color="#1f2933", linestyle="--", linewidth=1.4)
    ax.axhline(50, color="#9aa6b2", linestyle=":", linewidth=1)
    ax.scatter([median], [50], color="#1f2933", zorder=3)
    ax.text(median + 0.15, 52, f"Median {median:.1f}%", va="bottom", fontsize=8.5)
    ax.text(p10 + 0.15, 12, f"90% of rules reduce by at least {p10:.1f}%", fontsize=8.5)
    ax.set_xlim(max(70, values.min() - 1), min(100, values.max() + 1))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Candidate-space reduction (%)")
    ax.set_ylabel("Share of rules at or below reduction (%)")
    ax.set_title(title)
    ax.grid(axis="both", color="#d9dee3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _select_specific_examples(benchmark: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    preferred_patterns = [
        ("fashion completion",
         "Clothing > Pants + Clothing > Shorts + Activewear → Clothing > Clothing Tops"),
        ("fashion completion",
         "Clothing > Shorts + Clothing > Swimwear + Activewear → Clothing > Clothing Tops"),
        ("beauty completion",
         "Makeup > Lip Makeup + Cosmetics → Makeup > Face Makeup"),
        ("beauty completion",
         "Cosmetics > Makeup > Makeup Finishing Sprays → Makeup > Face Makeup"),
        ("toy affinity",
         "Dolls, Playsets & Toy Figures → Toys"),
    ]

    chosen_rows: List[pd.Series] = []
    seen_rules: set = set()
    for theme, rule_text in preferred_patterns:
        matches = benchmark[benchmark["rule"] == rule_text]
        if matches.empty:
            continue
        row = matches.iloc[0].copy()
        row["example_theme"] = theme
        chosen_rows.append(row)
        seen_rules.add(str(row["rule"]))
        if len(chosen_rows) >= max_examples:
            break

    if len(chosen_rows) < max_examples:
        fallback = benchmark[
            (benchmark["n_consequent_tokens"] == 1)
            & (~benchmark["rule"].isin(seen_rules))
        ].sort_values(["score", "search_space_reduction_pct", "confidence"], ascending=False)
        seen_consequents = {str(r["consequent_labels"]) for r in chosen_rows}
        for _, row in fallback.iterrows():
            consequent = str(row["consequent_labels"])
            if consequent in seen_consequents:
                continue
            row = row.copy()
            row["example_theme"] = "high-scoring rule"
            chosen_rows.append(row)
            seen_consequents.add(consequent)
            if len(chosen_rows) >= max_examples:
                break

    examples = pd.DataFrame(chosen_rows)
    if examples.empty:
        raise RuntimeError("Could not select any concrete rule examples.")
    return examples


def _select_leaf_examples(benchmark: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    leaf = benchmark[
        (benchmark["n_consequent_tokens"] == 1)
        & benchmark["consequent_tokens"].str.startswith("L0|", na=False)
    ].copy()
    if leaf.empty:
        return leaf

    preferred_patterns = [
        ("leaf fashion",
         "Clothing > Pants > Jeans + Activewear > Activewear Sweatshirts & Hoodies + Clothing > Shorts → Activewear > Activewear Tops > T-Shirts"),
        ("leaf fashion",
         "Activewear > Activewear Sweatshirts & Hoodies > Hoodies + Clothing > Clothing Tops + Clothing > Shorts → Activewear > Activewear Tops > T-Shirts"),
        ("leaf beauty",
         "Makeup > Face Makeup > Highlighters & Luminizers + Cosmetics + Skin Care → Makeup > Face Makeup > Blushes & Bronzers"),
        ("leaf jewelry",
         "Apparel & Accessories > Jewelry > Earrings + Apparel & Accessories > Jewelry > Necklaces + Apparel & Accessories > Jewelry > Rings → Apparel & Accessories > Jewelry > Bracelets"),
        ("leaf jewelry",
         "Apparel & Accessories > Jewelry > Bracelets + Apparel & Accessories > Jewelry > Rings + Makeup → Apparel & Accessories > Jewelry > Earrings"),
    ]

    chosen_rows: List[pd.Series] = []
    seen_rules: set = set()
    for theme, rule_text in preferred_patterns:
        matches = leaf[leaf["rule"] == rule_text]
        if matches.empty:
            continue
        row = matches.iloc[0].copy()
        row["example_theme"] = theme
        chosen_rows.append(row)
        seen_rules.add(str(row["rule"]))
        if len(chosen_rows) >= max_examples:
            break

    if len(chosen_rows) < max_examples:
        fallback = leaf[~leaf["rule"].isin(seen_rules)].sort_values(
            ["score", "search_space_reduction_pct", "confidence"], ascending=False
        )
        seen_consequents = {str(r["consequent_labels"]) for r in chosen_rows}
        for _, row in fallback.iterrows():
            consequent = str(row["consequent_labels"])
            if consequent in seen_consequents and len(chosen_rows) >= 3:
                continue
            row = row.copy()
            row["example_theme"] = "leaf rule"
            chosen_rows.append(row)
            seen_consequents.add(consequent)
            if len(chosen_rows) >= max_examples:
                break

    return pd.DataFrame(chosen_rows)


def _enrich_specific_examples(examples: pd.DataFrame, universe_size: int) -> pd.DataFrame:
    enriched = examples.copy()
    enriched["catalogue_products_before_rule"] = universe_size
    enriched["products_removed_from_search"] = (
        enriched["catalogue_products_before_rule"] - enriched["consequent_candidate_products"]
    )
    cols = [
        "example_theme", "rule",
        "antecedent_labels", "consequent_labels",
        "catalogue_products_before_rule", "consequent_candidate_products",
        "products_removed_from_search", "search_space_reduction_pct",
        "support", "confidence", "lift", "score",
    ]
    return enriched[cols]


def _shorten_rule_label(rule: str) -> str:
    replacements = {
        "Clothing > ": "",
        "Activewear > ": "",
        "Apparel & Accessories > Jewelry > ": "",
        "Makeup > ": "",
        "Cosmetics > ": "",
        "Dolls, Playsets & Toy Figures": "Dolls / Playsets",
    }
    label = str(rule)
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def _wrap_rule_label(rule: str, width: int = 34) -> str:
    return "\n".join(
        textwrap.wrap(
            _shorten_rule_label(rule),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _save_specific_examples_chart(
    examples: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    plot_df = examples.copy()
    plot_df["short_rule"] = plot_df["rule"].apply(_wrap_rule_label)
    plot_df = plot_df.sort_values("consequent_candidate_products", ascending=True).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["consequent_candidate_products"], color="#2f6f6d")
    xmax = float(plot_df["consequent_candidate_products"].max()) * 1.38
    ax.set_xlim(0, xmax)
    ax.set_yticks(y, plot_df["short_rule"])
    ax.tick_params(axis="y", labelsize=10, pad=8, length=0)
    ax.set_xlabel("Remaining candidate products after rule")
    ax.set_title(title)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#d9dee3", linewidth=0.8)

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        after     = int(row.consequent_candidate_products)
        reduction = float(row.search_space_reduction_pct)
        ax.text(after + xmax * 0.02, idx, f"{after:,} ({reduction:.1f}% fewer)",
                va="center", fontsize=9)

    baseline = int(plot_df["catalogue_products_before_rule"].iloc[0])
    fig.text(
        0.99, 0.01,
        f"Baseline categorized catalogue: {baseline:,} products",
        ha="right", va="bottom", fontsize=9, color="#59636e",
    )
    fig.subplots_adjust(left=0.43, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_rule_candidate_space(args: argparse.Namespace, out_dir: Path) -> None:
    catalogue_base = Path(args.catalogue_base or args.base)
    rules_csv      = Path(args.candidate_rules_csv)
    k              = args.catalogue_k_levels

    print(f"\n{'='*60}")
    print("EXPERIMENT 7: Rule-based candidate-space reduction")
    print(f"  rules CSV  : {rules_csv}")
    print(f"  catalogue  : {catalogue_base}")
    print(f"  K          : {k}")
    print(f"{'='*60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)

    with timed_step("Loading catalogue"):
        catalogue      = _load_catalogue(catalogue_base)
        universe_size  = int(catalogue["product_id"].nunique())
        token_products = _build_token_products(catalogue, k)
        inventory      = _build_category_inventory(token_products, universe_size)

    rules     = pd.read_csv(rules_csv)
    benchmark = _build_rule_benchmark(rules, token_products, universe_size)
    summary   = _build_candidate_summary(catalogue, inventory, benchmark)

    specific_examples_raw = _select_specific_examples(benchmark, args.candidate_specific_examples)
    specific_examples     = _enrich_specific_examples(specific_examples_raw, universe_size)

    leaf_examples_raw = _select_leaf_examples(benchmark, args.candidate_specific_examples)
    leaf_examples     = (
        _enrich_specific_examples(leaf_examples_raw, universe_size)
        if not leaf_examples_raw.empty
        else leaf_examples_raw
    )

    inventory.to_csv(out_dir / "category_inventory.csv", index=False)
    benchmark.to_csv(out_dir / "rule_candidate_space_reduction.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    top_examples = benchmark[benchmark["n_consequent_tokens"] == 1].copy()
    top_examples = top_examples.sort_values(
        ["score", "search_space_reduction_pct", "confidence"], ascending=False
    ).head(args.candidate_top_examples)
    top_examples.to_csv(out_dir / "top_rule_examples.csv", index=False)

    specific_examples.to_csv(out_dir / "specific_rule_examples.csv", index=False)
    if not leaf_examples.empty:
        leaf_examples.to_csv(out_dir / "leaf_rule_examples.csv", index=False)

    _save_category_histogram(inventory, out_dir / "category_unique_products_hist.png")
    _save_category_histogram(inventory, out_dir / "category_inventory_histogram.png")
    _save_rule_reduction_curve(
        benchmark,
        out_dir / "rule_candidate_space_reduction_hist.png",
        title="Rules sharply narrow the downstream candidate space",
    )
    single_consequent = benchmark[benchmark["n_consequent_tokens"] == 1]
    _save_rule_reduction_curve(
        single_consequent,
        out_dir / "single_consequent_rule_reduction_hist.png",
        title="Candidate-space reduction from rule consequents",
    )
    _save_rule_reduction_curve(
        single_consequent,
        out_dir / "candidate_space_reduction_from_rule_consequents.png",
        title="Candidate-space reduction from rule consequents",
    )
    _save_specific_examples_chart(
        specific_examples,
        out_dir / "specific_rule_examples_before_after.png",
        title="Concrete rules reduce the downstream candidate set",
    )
    if not leaf_examples.empty:
        _save_specific_examples_chart(
            leaf_examples,
            out_dir / "leaf_rule_examples_before_after.png",
            title="Leaf-level rules also narrow the candidate set",
        )

    row = summary.iloc[0]
    print(f"Categorized catalogue products: {int(row['categorized_products']):,}")
    print(f"Hierarchy-aware category tokens: {int(row['category_tokens']):,}")
    print(f"Rules benchmarked: {int(row['rules']):,}")
    print(
        f"Median candidate-space reduction: "
        f"{row['median_search_space_reduction_pct']:.1f}% "
        f"(IQR {row['p25_search_space_reduction_pct']:.1f}%–"
        f"{row['p75_search_space_reduction_pct']:.1f}%)"
    )
    print(
        f"Single-consequent median reduction: "
        f"{row['single_consequent_median_search_space_reduction_pct']:.1f}%"
    )
    print(f"Saved outputs to {out_dir}/")


# ===========================================================================
# EXPERIMENT 8 — Held-out recall evaluation (Figure 12)
# ===========================================================================

def _build_test_category_cases(
    test_rows: pd.DataFrame,
    k_levels: int,
    train_vocab: set,
) -> List[tuple]:
    """Build (observed_tokens, target_tokens) pairs from the held-out test wishlists.

    Each wishlist is split: all but the last category = observed context;
    the last category = target to predict. Only wishlists with ≥2 categories
    and non-empty context are kept.
    """
    rows = test_rows[["wishlist_id", "category_name"]].drop_duplicates().copy()
    rows["tokens"] = rows["category_name"].map(
        lambda path: frozenset(
            tok for tok in make_level_tokens(path, k_levels)
            if tok in train_vocab
        )
    )
    cases: List[tuple] = []
    for _, group in rows.groupby("wishlist_id"):
        records = sorted(group.itertuples(index=False), key=lambda r: str(r.category_name))
        if len(records) < 2:
            continue
        target = records[-1]
        observed = frozenset().union(*(r.tokens for r in records[:-1]))
        if observed and target.tokens:
            cases.append((observed, target.tokens))
    return cases


def _token_category_masks(token_products: pd.DataFrame, universe_size: int) -> Dict[str, int]:
    """Build integer bitmasks: one bit per unique product per category token."""
    product_ids = sorted(token_products["product_id"].unique(), key=str)
    product_index = {pid: i for i, pid in enumerate(product_ids)}
    masks: Dict[str, int] = {}
    for token, group in token_products.groupby("token"):
        mask = 0
        for pid in group["product_id"]:
            mask |= 1 << product_index[pid]
        masks[token] = mask
    return masks


def _evaluate_category_recall(
    cases: List[tuple],
    rules: pd.DataFrame,
    token_masks: Dict[str, int],
    universe_size: int,
    popular_tokens: List[str],
) -> Dict[str, float]:
    """Evaluate held-out category recall and candidate-space reduction.

    Returns a dict with reduction_pct and rule_recall_pct.
    """
    antecedent_index: Dict[frozenset, set] = {}
    for row in rules.itertuples():
        antecedent_index.setdefault(frozenset(row.a_toks), set()).add(row.b_toks[0])
    max_ante = max((len(k) for k in antecedent_index), default=0)

    hits, total, share_sum = 0, 0, 0.0
    for observed, target_tokens in cases:
        predicted: set = set()
        obs_list = sorted(observed)
        n_subsets = sum(
            math_comb(len(obs_list), sz)
            for sz in range(1, min(max_ante, len(obs_list)) + 1)
        )
        if n_subsets <= len(antecedent_index):
            for sz in range(1, min(max_ante, len(obs_list)) + 1):
                for subset in itertools_combinations(obs_list, sz):
                    predicted.update(antecedent_index.get(frozenset(subset), set()))
        else:
            for ante, cons in antecedent_index.items():
                if ante.issubset(observed):
                    predicted.update(cons)
        mask = 0
        for tok in predicted:
            mask |= token_masks.get(tok, 0)
        candidate_size = mask.bit_count()
        share_sum += 100.0 * candidate_size / universe_size if universe_size else 0.0
        if bool(predicted & set(target_tokens)):
            hits += 1
        total += 1

    if total == 0:
        return {"reduction_pct": 0.0, "rule_recall_pct": 0.0}
    mean_share = share_sum / total
    return {
        "reduction_pct": 100.0 - mean_share,
        "rule_recall_pct": 100.0 * hits / total,
    }


def _save_tradeoff_plot(
    tradeoff: pd.DataFrame,
    final_conf: float,
    out_path: Path,
) -> None:
    """Generate Figure 12: recall-reduction trade-off curve."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        tradeoff["reduction_pct"], tradeoff["held_out_recall_pct"],
        marker="o", linewidth=2.2, color="#2f6f6d",
    )
    offsets = {
        0.50: (0.15, 0.20),
        0.55: (0.15, 0.20),
        0.60: (0.20, -0.55),
        0.65: (0.15, 0.20),
        0.70: (0.15, 0.80),    # staggered vertically to avoid crowding near y=0
        0.75: (0.15, 1.40),
        0.80: (0.15, 2.00),
    }
    for row in tradeoff.itertuples(index=False):
        dx, dy = offsets.get(round(row.confidence_threshold, 2), (0.15, 0.15))
        ax.text(
            row.reduction_pct + dx, row.held_out_recall_pct + dy,
            f"conf >= {row.confidence_threshold:.2f}", fontsize=8,
        )
    rec = tradeoff.iloc[(tradeoff["confidence_threshold"] - final_conf).abs().argmin()]
    ax.scatter([rec["reduction_pct"]], [rec["held_out_recall_pct"]], s=80, color="#b23a48", zorder=3)
    ax.annotate(
        "Current final-rule setup",
        xy=(rec["reduction_pct"], rec["held_out_recall_pct"]),
        xytext=(rec["reduction_pct"] + 2.5, rec["held_out_recall_pct"] + 2.5),
        arrowprops={"arrowstyle": "->", "color": "#59636e"},
        fontsize=8.5,
    )
    ax.set_xlabel("Mean candidate-space reduction (%)")
    ax.set_ylabel("Held-out category recall (%)")
    ax.set_title(
        "Recall-reduction trade-off for rule-based candidate generation",
        fontsize=13, pad=10,
    )
    ax.tick_params(labelsize=10)
    ax.grid(color="#d9dee3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _compute_generalisation_metrics(
    test_rows: pd.DataFrame,
    rules: pd.DataFrame,
    k_levels: int,
    train_vocab: set,
) -> Dict[str, Any]:
    """Compute transaction-level held-out generalisation metrics.

    A rule *fires* on a test transaction when its full antecedent is a subset
    of the transaction's category tokens.  It *hits* when it fires and the
    single consequent token is also present in that transaction.

    Unlike the leave-one-out category-recall sweep, this function evaluates
    every rule against the *full* token set of each test wishlist — the same
    evaluation described in the thesis held-out table.

    Returns a dict with keys:
        n_rules               — rules evaluated
        n_rules_firing        — rules that fire on ≥1 test transaction
        rules_firing_pct      — percentage of rules that fire
        test_coverage_pct     — % of test transactions covered by ≥1 rule
        mean_hit_rate         — mean per-rule hit rate (firing rules only)
        pooled_hit_rate       — aggregate hits / aggregate firings
        n_test_transactions   — test transactions evaluated
    """
    if rules.empty:
        return {
            "n_rules": 0,
            "n_rules_firing": 0,
            "rules_firing_pct": 0.0,
            "test_coverage_pct": 0.0,
            "mean_hit_rate": 0.0,
            "pooled_hit_rate": 0.0,
            "n_test_transactions": 0,
        }

    # Build full token set per test wishlist (all categories; no leave-one-out)
    rows = test_rows[["wishlist_id", "category_name"]].drop_duplicates().copy()
    rows["tokens"] = rows["category_name"].map(
        lambda path: frozenset(
            tok for tok in make_level_tokens(path, k_levels)
            if tok in train_vocab
        )
    )
    test_transactions: List[frozenset] = []
    for _wid, group in rows.groupby("wishlist_id"):
        all_toks: frozenset = frozenset().union(*group["tokens"])
        if all_toks:
            test_transactions.append(all_toks)

    n_test = len(test_transactions)
    if n_test == 0:
        return {
            "n_rules": len(rules),
            "n_rules_firing": 0,
            "rules_firing_pct": 0.0,
            "test_coverage_pct": 0.0,
            "mean_hit_rate": 0.0,
            "pooled_hit_rate": 0.0,
            "n_test_transactions": 0,
        }

    # Pre-extract (antecedent frozenset, consequent token) for every rule
    rule_list: List[tuple] = [
        (frozenset(row.a_toks), row.b_toks[0])
        for row in rules.itertuples()
    ]
    n_rules = len(rule_list)
    rule_firings: List[int] = [0] * n_rules
    rule_hits:    List[int] = [0] * n_rules

    # Single pass over test transactions — O(n_test × n_rules × |antecedent|)
    n_covered = 0
    for txn in test_transactions:
        any_fired = False
        for i, (ante, cons) in enumerate(rule_list):
            if ante.issubset(txn):
                rule_firings[i] += 1
                any_fired = True
                if cons in txn:
                    rule_hits[i] += 1
        if any_fired:
            n_covered += 1

    firing_idx = [i for i, f in enumerate(rule_firings) if f > 0]
    n_rules_firing = len(firing_idx)

    mean_hit_rate = (
        sum(rule_hits[i] / rule_firings[i] for i in firing_idx) / n_rules_firing
        if n_rules_firing > 0 else 0.0
    )
    total_firings = sum(rule_firings[i] for i in firing_idx)
    total_hits    = sum(rule_hits[i]    for i in firing_idx)
    pooled_hit_rate = total_hits / total_firings if total_firings > 0 else 0.0

    return {
        "n_rules":             n_rules,
        "n_rules_firing":      n_rules_firing,
        "rules_firing_pct":    100.0 * n_rules_firing / n_rules,
        "test_coverage_pct":   100.0 * n_covered / n_test,
        "mean_hit_rate":       mean_hit_rate,
        "pooled_hit_rate":     pooled_hit_rate,
        "n_test_transactions": n_test,
    }


def run_held_out_recall(args: argparse.Namespace, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("EXPERIMENT 8: Held-out recall evaluation  (Figure 12)")
    cat_base = Path(getattr(args, "catalogue_base", None) or args.base)
    print(f"  Catalogue base : {cat_base}")
    print(f"  Train fraction : {args.held_out_train_frac}")
    print(f"  Mine min conf  : {args.held_out_mine_conf}")
    print(f"  Conf sweep     : {args.held_out_conf_sweep}")
    print(f"{'='*60}\n")

    _cat_ok = (
        ((cat_base / "products").is_dir() and (cat_base / "categories").is_dir())
        or (cat_base.is_dir() and bool(list(cat_base.glob("*.parquet"))))
        or (cat_base.is_file() and cat_base.suffix == ".parquet")
    )
    if not _cat_ok:
        print(
            "  [EXPERIMENT 8 skipped] Catalogue not found at "
            f"{cat_base}. Pass --catalogue-base to supply it."
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x.strip()) for x in args.held_out_conf_sweep.split(",")]

    # ── Load wishlist rows ────────────────────────────────────────────────────
    print(f"  Loading wishlist data from {args.base} …")
    with timed_step("Loading data"):
        df_raw = load_data(args.base)
    with timed_step("Cleaning data"):
        df_min = clean_data(df_raw)
    if args.sample:
        keep = df_min["wishlist_id"].drop_duplicates().head(args.sample)
        df_min = df_min[df_min["wishlist_id"].isin(set(keep))]

    all_ids = df_min["wishlist_id"].drop_duplicates().to_numpy()
    n_train = int(len(all_ids) * args.held_out_train_frac)
    train_ids = set(all_ids[:n_train])
    test_ids  = set(all_ids[n_train:])
    print(f"  Wishlists — train: {len(train_ids):,}  test: {len(test_ids):,}")

    train_rows = df_min[df_min["wishlist_id"].isin(train_ids)].copy()
    test_rows  = df_min[df_min["wishlist_id"].isin(test_ids)].copy()

    # ── Load catalogue ────────────────────────────────────────────────────────
    print(f"  Loading catalogue from {cat_base} …")
    with timed_step("Loading catalogue"):
        catalogue = _load_catalogue(cat_base)
    universe_size = int(catalogue["product_id"].nunique())
    print(f"  Catalogue: {universe_size:,} categorized products")

    token_products = _build_token_products(catalogue, args.catalogue_k_levels)
    token_masks = _token_category_masks(token_products, universe_size)

    # ── Mine rules on training split ─────────────────────────────────────────
    print("  Mining rules on training split …")
    with timed_step("Building train transactions"):
        train_txns, train_vocab = build_transactions(train_rows, args.catalogue_k_levels)

    path_map = {tok: label_of(tok) for tok in train_vocab}

    with timed_step("Building ancestor map"):
        branch_ancestry = _build_branch_ancestry_from_category_paths(train_rows)
        ancestors = build_ancestors_from_tokens(train_vocab, branch_ancestry=branch_ancestry)

    with timed_step("Encoding transactions"):
        df_encoded = encode_transactions(train_txns, train_vocab)

    with timed_step("Mining raw rules"):
        rules_raw = mine_rules_raw(
            df_encoded,
            min_support=args.ksweep_support,
            min_conf=args.held_out_mine_conf,
            max_len=6,
            ancestors=ancestors,
            path_map=path_map,
            branch_ancestry=branch_ancestry,
            candidate_pruning="ancestor-same-path",
            rule_filtering="full",
        )

    rules_raw["a_toks"] = rules_raw["antecedents"].apply(lambda s: tuple(sorted(s)))
    rules_raw["b_toks"] = rules_raw["consequents"].apply(lambda s: tuple(sorted(s)))
    mined_rules = add_score_and_rank(rules_raw, min_support=args.ksweep_support, min_lift=args.ksweep_lift)
    mined_rules = mined_rules[mined_rules["lift"] >= args.ksweep_lift].copy()
    mined_rules = dedupe_family(mined_rules)
    mined_rules = dedupe_antimirror(mined_rules)
    print(f"  Mined {len(mined_rules):,} rules on train split at conf≥{args.held_out_mine_conf}")

    # ── Build test cases ──────────────────────────────────────────────────────
    with timed_step("Building test cases"):
        test_cases = _build_test_category_cases(test_rows, args.catalogue_k_levels, set(train_vocab))
    print(f"  Test cases: {len(test_cases):,}")

    # ── Sweep confidence thresholds ───────────────────────────────────────────
    popular_tokens = (
        pd.Series([tok for txn in train_txns for tok in txn])
        .value_counts()
        .index.tolist()
    )

    tradeoff_rows: List[Dict[str, Any]] = []
    for conf in thresholds:
        subset = mined_rules[
            (mined_rules["confidence"] >= conf) &
            (mined_rules["b_toks"].map(len) == 1)
        ].copy()
        metrics = _evaluate_category_recall(
            test_cases, subset, token_masks, universe_size, popular_tokens
        )
        tradeoff_rows.append({
            "confidence_threshold": conf,
            "rules": len(subset),
            "reduction_pct": metrics["reduction_pct"],
            "held_out_recall_pct": metrics["rule_recall_pct"],
        })
        print(
            f"  conf≥{conf:.2f}  rules={len(subset):,}  "
            f"reduction={metrics['reduction_pct']:.1f}%  "
            f"recall={metrics['rule_recall_pct']:.1f}%"
        )

    tradeoff = pd.DataFrame(tradeoff_rows)
    tradeoff.to_csv(out_dir / "recall_reduction_tradeoff.csv", index=False)

    _save_tradeoff_plot(
        tradeoff,
        args.held_out_final_conf,
        out_dir / "recall_reduction_tradeoff.png",
    )

    # ── Generalisation metrics at the final confidence threshold ─────────────
    final_subset = mined_rules[
        (mined_rules["confidence"] >= args.held_out_final_conf) &
        (mined_rules["b_toks"].map(len) == 1)
    ].copy()

    gen = _compute_generalisation_metrics(
        test_rows=test_rows,
        rules=final_subset,
        k_levels=args.catalogue_k_levels,
        train_vocab=set(train_vocab),
    )

    gen_row = {
        "confidence_threshold":  args.held_out_final_conf,
        "n_train_transactions":  len(train_ids),
        "n_test_transactions":   gen["n_test_transactions"],
        "n_rules":               gen["n_rules"],
        "n_rules_firing":        gen["n_rules_firing"],
        "rules_firing_pct":      gen["rules_firing_pct"],
        "test_coverage_pct":     gen["test_coverage_pct"],
        "mean_hit_rate":         gen["mean_hit_rate"],
        "pooled_hit_rate":       gen["pooled_hit_rate"],
    }
    pd.DataFrame([gen_row]).to_csv(out_dir / "held_out_generalisation.csv", index=False)

    print(
        f"\n  Generalisation metrics (conf≥{args.held_out_final_conf}):\n"
        f"    Rules mined on train  : {gen['n_rules']:,}\n"
        f"    Rules firing on test  : {gen['n_rules_firing']:,} "
        f"({gen['rules_firing_pct']:.1f}%)\n"
        f"    Test-wishlist coverage: {gen['test_coverage_pct']:.1f}%\n"
        f"    Mean per-rule hit rate: {gen['mean_hit_rate']:.3f}\n"
        f"    Pooled hit rate       : {gen['pooled_hit_rate']:.3f}"
    )

    print(f"\n  Saved outputs to {out_dir}/")


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
        "l0_pair_example", "rule_candidate_space",
        "held_out_recall",
    } - skip

    out_root = Path(args.output_dir).resolve()
    requested_sample = args.sample
    if requested_sample is not None:
        args.mining_base = str(_materialize_wishlist_sample(args.base, requested_sample, out_root))
        args.sample = None
    elif args.mining_base is None:
        args.mining_base = args.base

    print(f"[{now_stamp()}] Benchmark started")
    print(f"Data path    : {args.base}")
    print(f"Mining data  : {args.mining_base}")
    print(f"Output dir   : {out_root}")
    print(f"Running      : {', '.join(sorted(run))}")
    if skip:
        print(f"Skipping     : {', '.join(sorted(skip))}")
    if requested_sample is not None:
        print(f"Sample       : {requested_sample} wishlists")
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

    # Experiments 6, 7, 8 require a product catalogue.  Supported formats:
    #   - Old format: products/ and categories/ parquet sub-directories
    #   - New format: merged parquet(s) with wishlist_id, product_id, category_name
    # When --catalogue-base is not supplied we fall back to the sample dir (base).
    _cat_base = Path(args.catalogue_base or args.base)
    _has_catalogue = (
        ((_cat_base / "products").is_dir() and (_cat_base / "categories").is_dir())
        or (_cat_base.is_dir() and bool(list(_cat_base.glob("*.parquet"))))
        or (_cat_base.is_file() and _cat_base.suffix == ".parquet")
    )
    _catalogue_hint = (
        f"Provide --catalogue-base pointing to the merged parquet file or directory, "
        f"or a directory with products/ and categories/ sub-directories."
    )

    if "l0_pair_example" in run:
        if not _has_catalogue:
            print(
                f"\n[EXPERIMENT 6 skipped] Catalogue not found at {_cat_base}. "
                + _catalogue_hint
            )
        else:
            run_l0_pair_example(args, out_root / "l0_pair_example")

    if "rule_candidate_space" in run:
        candidate_args = argparse.Namespace(**vars(args))
        if not candidate_args.candidate_rules_csv:
            example_rules_csv = (
                out_root / "example_rules" /
                f"rules_k{args.example_k}_s{str(args.example_support).replace('.','p')}.csv"
            )
            if example_rules_csv.exists():
                candidate_args.candidate_rules_csv = str(example_rules_csv)
                if candidate_args.catalogue_k_levels != args.example_k:
                    print(
                        f"\n[EXPERIMENT 7] Aligning --catalogue-k-levels "
                        f"{candidate_args.catalogue_k_levels} → {args.example_k} because the "
                        f"auto-selected rules were mined with K={args.example_k}."
                    )
                    candidate_args.catalogue_k_levels = args.example_k
                print(
                    f"\n[EXPERIMENT 7] Using generated example-rules CSV: "
                    f"{candidate_args.candidate_rules_csv}"
                )
            else:
                print(
                    "\n[EXPERIMENT 7 skipped] No candidate rules CSV found. "
                    "Run Experiment 3 in the same benchmark invocation, or pass "
                    "--candidate-rules-csv explicitly."
                )
        if candidate_args.candidate_rules_csv and not _has_catalogue:
            print(
                f"\n[EXPERIMENT 7 skipped] Catalogue not found at {_cat_base}. "
                + _catalogue_hint
            )
        elif candidate_args.candidate_rules_csv:
            run_rule_candidate_space(candidate_args, out_root / "rule_candidate_space")

    if "held_out_recall" in run:
        if not _has_catalogue:
            print(
                f"\n[EXPERIMENT 8 skipped] Catalogue not found at {_cat_base}. "
                + _catalogue_hint
            )
        else:
            run_held_out_recall(args, out_root / "held_out_recall")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[{now_stamp()}] Benchmark finished in {format_elapsed(total_elapsed)}")
    print(f"All results under {out_root}/")


if __name__ == "__main__":
    main()
