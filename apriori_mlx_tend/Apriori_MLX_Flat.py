"""
Association-rule mining pipeline — flat mlxtend Apriori.

Usage:
    python Apriori_MLX_Flat.py <base> [options]

`base` is a directory containing the pre-merged parquet(s)
(columns: wishlist_id, category_name).

Each wishlist is one transaction; items are category_name strings at the
chosen taxonomy depth (--level).  The pipeline then runs mlxtend Apriori
exactly as documented at:
  https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
"""

import argparse
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.3f}s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)}m {rem:.1f}s"


@contextmanager
def timed_step(label: str):
    start = time.perf_counter()
    print(f"[{now_stamp()}] {label} …", flush=True)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        print(f"[{now_stamp()}] {label} FAILED after {format_elapsed(elapsed)}", flush=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        print(f"[{now_stamp()}] {label} done in {format_elapsed(elapsed)}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_optional_int(val) -> Optional[int]:
    if val is None or (isinstance(val, str) and val.lower() == "none"):
        return None
    return int(val)


LevelSpec = Union[int, str]


def _level_type(v: str) -> LevelSpec:
    if v in ("leaf-path", "full-path", "full"):
        return "leaf_path"
    if v in ("leaf", "-1"):
        return -1
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid level '{v}'. Use an integer or 'leaf'.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Mine association rules from pre-merged wishlist parquet (mlxtend Apriori).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("base",
                   help="Directory containing pre-merged parquet(s) "
                        "(columns: wishlist_id, category_name).")
    p.add_argument("--level", type=_level_type, default=0,
                   help="Taxonomy depth of category to use as item "
                        "(0=root, 1=second, ..., leaf/-1=deepest label, "
                        "leaf-path/full-path=unique full leaf path).")
    p.add_argument("--min-support",  type=float, default=0.01, metavar="S")
    p.add_argument("--min-conf",     type=float, default=0.3,  metavar="C")
    p.add_argument("--min-lift",     type=float, default=1.0,  metavar="L")
    p.add_argument("--max-len",      default=None, metavar="M",
                   help="Max itemset length passed to apriori() (int or 'none').")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Save rules to CSV (prints to stdout if omitted).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data root resolution
# ---------------------------------------------------------------------------

def resolve_root(cli_path: str) -> Path:
    p = Path(cli_path)
    if not p.is_dir():
        sys.exit(f"Path not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_data(base: Path) -> pd.DataFrame:
    if base.is_dir():
        files = sorted(base.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {base}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(base)
    print(f"  rows: {len(df):,}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["wishlist_id", "category_name"])
    print(f"  Dropped {before - len(df):,} duplicates → {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Category extraction at a given taxonomy level
# ---------------------------------------------------------------------------

def _extract_level(category_name: str, level: LevelSpec) -> Optional[str]:
    parts = [p.strip() for p in category_name.split(">") if p.strip()]
    if not parts:
        return None
    if level == "leaf_path":
        return " > ".join(parts)
    if level == -1:
        return parts[-1]
    return parts[level] if level < len(parts) else None


# ---------------------------------------------------------------------------
# Build transactions & encode
# ---------------------------------------------------------------------------

def build_transactions(df: pd.DataFrame, level: LevelSpec) -> Tuple[List[List[str]], int]:
    df = df.copy()
    df["item"] = df["category_name"].apply(lambda s: _extract_level(s, level))
    df = df.dropna(subset=["item"])
    df = df.drop_duplicates(subset=["wishlist_id", "item"])

    baskets = df.groupby("wishlist_id")["item"].apply(list).tolist()
    n_multi = sum(1 for b in baskets if len(b) >= 2)

    print(f"  Total baskets        : {len(baskets):,}")
    print(f"  Baskets with ≥2 items: {n_multi:,}")
    print(f"  Unique items         : {df['item'].nunique():,}")

    return baskets, n_multi


def encode(baskets: List[List[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    X = te.fit(baskets).transform(baskets, sparse=True)
    return pd.DataFrame.sparse.from_spmatrix(X, columns=te.columns_)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.perf_counter()
    args = parse_args()
    args.max_len = _parse_optional_int(args.max_len)

    base = resolve_root(args.base)

    print(f"[{now_stamp()}] Pipeline started")
    print(f"Base path   : {base}")
    print(f"level       : {args.level}")
    print(f"min_support : {args.min_support}")
    print(f"min_conf    : {args.min_conf}")
    print(f"min_lift    : {args.min_lift}")
    print(f"max_len     : {args.max_len}")
    print()

    with timed_step("Loading data"):
        df = load_data(base)

    with timed_step("Cleaning data"):
        df = clean_data(df)

    with timed_step("Building transactions"):
        baskets, _ = build_transactions(df, args.level)

    with timed_step("Encoding (TransactionEncoder)"):
        df_encoded = encode(baskets)

    with timed_step("Apriori"):
        freq = apriori(
            df_encoded,
            min_support=args.min_support,
            use_colnames=True,
            max_len=args.max_len,
            low_memory=False,
        )
        print(f"  Frequent itemsets: {len(freq):,}")

    with timed_step("Association rules"):
        rules = association_rules(
            freq,
            num_itemsets=len(baskets),
            metric="lift",
            min_threshold=args.min_lift,
        )
        rules = rules[rules["confidence"] >= args.min_conf]
        rules = rules.sort_values("lift", ascending=False)
        print(f"  Rules: {len(rules):,}")

    print(f"\nFound {len(rules):,} rules.\n")

    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(s)))

    if args.output:
        with timed_step("Saving"):
            out.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 160)
        print(out[cols].to_string(index=False))

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[{now_stamp()}] Done in {format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
