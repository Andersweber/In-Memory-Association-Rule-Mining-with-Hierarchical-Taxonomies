#!/usr/bin/env python3
"""
create_samples.py — Generate wishlist samples from the merged parquet in Data/.

Place your merged parquet file(s) in the Data/ folder
(columns: wishlist_id, category_name), then run this script to produce
size-limited samples used by the benchmarks.

Usage:
    python create_samples.py [--sizes 5000 20000 50000 100000] [--overwrite]

Output structure:
    Data/samples/
        5000/
        20000/
        50000/
        100000/

Each subfolder is a valid parquet directory readable by pd.read_parquet().
Pass a subfolder path to any pipeline script or to ProductionBenchmark.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT      = Path(__file__).resolve().parent
DATA_DIR       = REPO_ROOT / "Data"
DEFAULT_OUTPUT = DATA_DIR / "samples"
DEFAULT_SIZES  = [5_000, 20_000, 50_000, 100_000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create size-specific benchmark samples from the merged parquet in Data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
        help="Number of wishlists per sample.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recreate existing sample folders.",
    )
    return p.parse_args()


def write_parquet_dir(df: pd.DataFrame, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "part-00000.parquet", index=False, engine="pyarrow")


def main() -> None:
    args = parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"Data/ folder not found at {DATA_DIR}")

    print(f"Reading merged parquet from: {DATA_DIR}")
    df = pd.read_parquet(DATA_DIR, engine="pyarrow")

    required = {"wishlist_id", "category_name"}
    missing  = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Merged parquet is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    before = len(df)
    df = df.drop_duplicates(subset=["wishlist_id", "category_name"])
    print(f"Loaded {before:,} rows → {len(df):,} after dedup")

    all_wishlist_ids = df["wishlist_id"].drop_duplicates().tolist()
    print(f"Total unique wishlists: {len(all_wishlist_ids):,}")
    print()

    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        target = DEFAULT_OUTPUT / str(size)

        if target.exists() and not args.overwrite:
            print(f"Skipping existing sample: {target}  (use --overwrite to recreate)")
            continue

        if target.exists():
            shutil.rmtree(target)

        if size > len(all_wishlist_ids):
            print(
                f"Warning: requested {size:,} wishlists but only "
                f"{len(all_wishlist_ids):,} available — using all."
            )

        keep_ids  = set(all_wishlist_ids[:size])
        sample_df = df[df["wishlist_id"].isin(keep_ids)].copy()
        kept_ids  = sample_df["wishlist_id"].nunique()

        print(
            f"Writing sample '{size}': "
            f"{kept_ids:,} wishlists, {len(sample_df):,} rows → {target}"
        )
        write_parquet_dir(sample_df, target)

        (target / "sample_metadata.json").write_text(
            json.dumps({
                "requested_wishlists": size,
                "actual_wishlists":    int(kept_ids),
                "rows":                int(len(sample_df)),
                "source":              str(DATA_DIR),
            }, indent=2),
            encoding="utf-8",
        )

    print(f"\nDone. Samples are under {DEFAULT_OUTPUT}/")
    print("Pass a sample folder to any pipeline or to ProductionBenchmark.py:")
    print(f"  python ProductionBenchmark.py Data/samples/100000")


if __name__ == "__main__":
    main()
