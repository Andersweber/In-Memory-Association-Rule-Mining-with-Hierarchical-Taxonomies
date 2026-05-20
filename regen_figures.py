"""
Standalone figure regeneration script.
Reads existing benchmark CSVs and re-generates all thesis figures
using the updated plotting code from Benchmark.py.

Usage:
    python regen_figures.py
"""

import sys
import types
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Locate data
# ---------------------------------------------------------------------------

K_SWEEP_CSV      = Path("/Users/anderschristensen/Downloads/results/k_sweep/k_sweep_summary.csv")
SUPPORT_CSV      = Path("/Users/anderschristensen/Downloads/results/support_sweep/support_sweep_summary.csv")
K_OUT_DIR        = Path("/Users/anderschristensen/Downloads/results/k_sweep")
SUPPORT_OUT_DIR  = Path("/Users/anderschristensen/Downloads/results/support_sweep")

# ---------------------------------------------------------------------------
# Import the plotting functions from Benchmark.py (same directory)
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

# We need to import _make_k_sweep_figures and _make_support_sweep_figures.
# Benchmark.py uses a global rcParams block at import time, which is what we want.
import importlib.util
spec = importlib.util.spec_from_file_location("Benchmark", BENCH_DIR / "Benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

_make_k_sweep_figures     = bench._make_k_sweep_figures
_make_support_sweep_figures = bench._make_support_sweep_figures

# ---------------------------------------------------------------------------
# Build fake args namespace matching what the plotting functions need
# ---------------------------------------------------------------------------

class Args:
    ksweep_support = 0.02
    ksweep_conf    = 0.6
    ksweep_lift    = 1.5

args = Args()

# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------

print("Loading k_sweep CSV …")
k_rows = pd.read_csv(K_SWEEP_CSV).to_dict(orient="records")
# Mark success
for r in k_rows:
    r["success"] = bool(r.get("success", True))

print(f"  {len(k_rows)} rows, implementations: "
      f"{sorted({r.get('implementation') for r in k_rows})}")

print("Loading support_sweep CSV …")
s_rows = pd.read_csv(SUPPORT_CSV).to_dict(orient="records")
for r in s_rows:
    r["success"] = bool(r.get("success", True))

print(f"  {len(s_rows)} rows, supports: "
      f"{sorted({r.get('min_support') for r in s_rows})}")

# ---------------------------------------------------------------------------
# Regenerate k-sweep figures
# ---------------------------------------------------------------------------

print("\nGenerating k-sweep figures …")
_make_k_sweep_figures(k_rows, K_OUT_DIR, args)

# ---------------------------------------------------------------------------
# Regenerate support-sweep figures (pass k_sweep CSV path so C++ data
# from k_sweep can appear in the efficiency scatter if present)
# ---------------------------------------------------------------------------

print("\nGenerating support-sweep figures …")
_make_support_sweep_figures(s_rows, SUPPORT_OUT_DIR, args, k_sweep_csv=K_SWEEP_CSV)

print("\nDone.  Figures written to:")
print(f"  {K_OUT_DIR / 'figures'}")
print(f"  {SUPPORT_OUT_DIR / 'figures'}")
