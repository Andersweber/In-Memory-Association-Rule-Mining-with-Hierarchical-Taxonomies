#!/usr/bin/env bash
#SBATCH --job-name=assoc_bench
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/benchmark}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100000}"
REPEATS="${REPEATS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/benchmark_results/sample_${SAMPLE_SIZE}}"
RUN_FULL="${RUN_FULL:-0}"
REQUIRE_CPP="${REQUIRE_CPP:-1}"
FULL_BASE="${FULL_BASE:-}"

cd "$PROJECT_DIR"
mkdir -p slurm_logs benchmark_results

module load anaconda3/2024.10-py3.12.7 2>/dev/null || true
module load gcc/13.2.0 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

python - <<'PY'
import importlib.util
missing = [
    name for name in ("pandas", "pyarrow", "numpy", "scipy", "joblib", "mlxtend", "matplotlib")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + "\nInstall them with: python -m pip install --user -r requirements.txt"
    )
PY

if [ ! -d "Data/samples/$SAMPLE_SIZE" ]; then
  python create_samples.py --sizes "$SAMPLE_SIZE"
fi

CPP_EXE="$PROJECT_DIR/apriori_cumulate/cpp/apriori_cumulate_cpp"
PY_PREFIX="$(python - <<'PY'
import sys
print(sys.prefix)
PY
)"
CPP_PREFIX="${CONDA_PREFIX:-$PY_PREFIX}"
NEEDS_CPP_BUILD=0
if [ ! -x "$CPP_EXE" ]; then
  NEEDS_CPP_BUILD=1
elif command -v file >/dev/null 2>&1 && ! file "$CPP_EXE" | grep -Eq 'ELF|Linux'; then
  NEEDS_CPP_BUILD=1
fi

if [ "$NEEDS_CPP_BUILD" = "1" ]; then
  (
    cd apriori_cumulate/cpp
    make CONDA_ENV="$CPP_PREFIX" clean
    make CONDA_ENV="$CPP_PREFIX" apriori_cumulate_cpp
  )
fi

if [ "$REQUIRE_CPP" = "1" ] && [ ! -x "$CPP_EXE" ]; then
  echo "C++ executable was not built: $CPP_EXE" >&2
  exit 1
fi

if [ "$REQUIRE_CPP" = "1" ] && command -v file >/dev/null 2>&1 && ! file "$CPP_EXE" | grep -Eq 'ELF|Linux'; then
  echo "C++ executable is not a Linux binary: $(file "$CPP_EXE")" >&2
  exit 1
fi

args=(
  "Data/samples/$SAMPLE_SIZE"
  --output-dir "$OUTPUT_DIR"
  --repeats "$REPEATS"
)

if [ -x "$CPP_EXE" ]; then
  args+=(--cpp-exe "$CPP_EXE")
fi

if [ -n "$FULL_BASE" ]; then
  args+=(--catalogue-base "$FULL_BASE" --full-base "$FULL_BASE")
fi

if [ "$RUN_FULL" != "1" ]; then
  args+=(--skip l0_pair_example rule_candidate_space held_out_recall full_dataset)
fi

python Benchmark.py "${args[@]}"
