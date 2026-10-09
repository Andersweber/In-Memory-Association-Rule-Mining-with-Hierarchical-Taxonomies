#!/bin/bash
#SBATCH --job-name=assoc_bench
#SBATCH --output=slurm_logs/assoc_bench_%j.out
#SBATCH --error=slurm_logs/assoc_bench_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail
trap 'rc=$?; echo "BENCHMARK_JOB_EXIT=$rc"; exit "$rc"' EXIT
module load anaconda3/2024.10-py3.12.7
module load gcc/13.2.0
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "${PROJECT_DIR:-$HOME/benchmark}"
PY="${BENCH_PYTHON:-$HOME/benchmark_env/bin/python}"
PROFILE="${PROFILE:-smoke}"
REPEATS="${REPEATS:-1}"
RUN_FULL="${RUN_FULL:-0}"
REQUIRE_CPP="${REQUIRE_CPP:-1}"
[[ "$REQUIRE_CPP" == 1 ]] || { echo 'REQUIRE_CPP must be 1' >&2; exit 2; }
[[ "$RUN_FULL" =~ ^[01]$ ]] || { echo 'RUN_FULL must be 0 or 1' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be positive' >&2; exit 2; }
EXTRA=()
case "$PROFILE" in
  smoke)
    SAMPLE_SIZE="${SAMPLE_SIZE:-5000}"
    [[ "$RUN_FULL" == 0 ]] || { echo 'Smoke profile does not run Experiment 9' >&2; exit 2; }
    EXTRA=(--ksweep-k 1 3 --ksweep-support 0.05 --ssweep-support 0.05
           --skip sensitivity basic_vs_cumulate example_rules l0_pair_example
           rule_candidate_space held_out_recall full_dataset)
    ;;
  full)
    SAMPLE_SIZE="${SAMPLE_SIZE:-100000}"
    if [[ "$RUN_FULL" == 0 ]]; then EXTRA=(--skip full_dataset); fi
    ;;
  *) echo 'PROFILE must be smoke or full' >&2; exit 2 ;;
esac
[[ "$SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'SAMPLE_SIZE must be positive' >&2; exit 2; }

BASE="$PWD/Data/samples/$SAMPLE_SIZE"
CATALOGUE="$PWD/Data/wishlist_data.parquet"
CPP="$PWD/apriori_cumulate/cpp/apriori_cumulate_cpp"
test -x "$PY"
test -d "$BASE"
test -s "$CATALOGUE"
test -x "$CPP"
OUT="$PWD/benchmark_results/${PROFILE}_${SAMPLE_SIZE}_${SLURM_JOB_ID:?Submit with sbatch}"
mkdir -p benchmark_results
mkdir "$OUT"
echo "OUTPUT_DIRECTORY=$OUT"
file "$CPP" | tee "$OUT/binary.txt"
grep -Eq 'ELF .*executable' "$OUT/binary.txt"
ldd "$CPP" > "$OUT/ldd.txt"
if grep -q 'not found' "$OUT/ldd.txt"; then cat "$OUT/ldd.txt" >&2; exit 1; fi

"$PY" -u slurm/run_checked_benchmark.py "$BASE" \
  --output-dir "$OUT" --catalogue-base "$CATALOGUE" --full-base "$CATALOGUE" \
  --cpp-exe "$CPP" --python-exe "$PY" --repeats "$REPEATS" --basic-max-len 5 \
  "${EXTRA[@]}" 2>&1 | tee "$OUT/benchmark.log"
