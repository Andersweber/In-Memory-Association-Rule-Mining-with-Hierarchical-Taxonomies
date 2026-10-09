#!/bin/bash
#SBATCH --job-name=gowish-memory-clean
#SBATCH --output=slurm_logs/gowish-memory-clean_%j.out
#SBATCH --error=slurm_logs/gowish-memory-clean_%j.err
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -u
cd ~/benchmark || exit 1
module load anaconda3/2024.10-py3.12.7 2>/dev/null || true
module load gcc/13.2.0 2>/dev/null || true

PY=~/benchmark_env/bin/python
CONDA=/opt/software/anaconda3/2024.10-py3.12.7
OUT="thesis_memory_clean_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# C++ binary for this node's CPU, under its own name so the benchmark binary is untouched
g++ -std=c++17 -O3 -march=native -DMAX_TRANSACTIONS=1048576 -I"$CONDA/include" \
  apriori_cumulate/cpp/Apriori_Cumulate_CPP.cpp -o apriori_cumulate/cpp/apriori_cumulate_cpp_mem \
  -L"$CONDA/lib" -larrow -larrow_dataset -lparquet -Wl,-rpath,"$CONDA/lib" || exit 1
CPP=./apriori_cumulate/cpp/apriori_cumulate_cpp_mem

{
  echo "job=$SLURM_JOB_ID"; echo "node=$SLURM_JOB_NODELIST"; hostname
  lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Thread' || true
  echo "PY=$PY"; "$PY" --version
} | tee "$OUT/node_env.txt"

# Peak RSS per run: a fresh Python wrapper starts the command as its only child
cat > "$OUT/measure_one.py" <<'PYEND'
import json, resource, subprocess, sys, time
from pathlib import Path
label, out, cmd = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
d = out / label
d.mkdir(parents=True, exist_ok=True)
(d / "command.txt").write_text(" ".join(cmd) + "\n")
t0 = time.perf_counter()
with open(d / "stdout.log", "wb") as o, open(d / "stderr.log", "wb") as e:
    p = subprocess.run(cmd, stdout=o, stderr=e)
m = {"label": label, "status": p.returncode, "elapsed_s": round(time.perf_counter() - t0, 1),
     "max_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
(d / "metrics.json").write_text(json.dumps(m) + "\n")
print(json.dumps(m), flush=True)
PYEND

run_mem () { label="$1"; shift; "$PY" "$OUT/measure_one.py" "$label" "$OUT" "$@" | tee -a "$OUT/run_status.log"; }

S=Data/samples/100000
F=Data/wishlist_data.parquet
CUM=apriori_cumulate/python/Apriori_Cumulate_Python.py
ARGS="--min-support 0.02 --min-conf 0.6 --min-lift 1.5 --max-ante-len 3 --max-cons-len 2 --max-len 5"

run_mem python_100k_k1 "$PY" $CUM $S --k-levels 1 $ARGS --output "$OUT/python_100k_k1/rules.csv"
run_mem python_100k_k3 "$PY" $CUM $S --k-levels 3 $ARGS --output "$OUT/python_100k_k3/rules.csv"
run_mem python_100k_k5 "$PY" $CUM $S --k-levels 5 $ARGS --output "$OUT/python_100k_k5/rules.csv"
run_mem python_full_k3 "$PY" $CUM $F --k-levels 3 $ARGS --output "$OUT/python_full_k3/rules.csv"
run_mem mlxtend_flat_100k "$PY" apriori_mlx_tend/Apriori_MLX_Flat.py $S --level leaf --min-support 0.02 --min-conf 0.6 --min-lift 1.0 --output "$OUT/mlxtend_flat_100k/rules.csv"
run_mem cpp_100k_k3 $CPP $S 3 0.02 0.6 1.5 3 2 "$OUT/cpp_100k_k3/rules.csv"
run_mem cpp_full_k3 $CPP $F 3 0.02 0.6 1.5 3 2 "$OUT/cpp_full_k3/rules.csv"
run_mem mlxtend_basic_100k_s0p03 "$PY" apriori_mlx_ancestor/Apriori_MLX_Ancestor.py $S --mode basic --k-levels 3 --min-support 0.03 --min-conf 0.6 --min-lift 1.5 --output "$OUT/mlxtend_basic_100k_s0p03/rules.csv"
run_mem mlxtend_basic_100k_s0p02_expected_fail "$PY" apriori_mlx_ancestor/Apriori_MLX_Ancestor.py $S --mode basic --k-levels 3 --min-support 0.02 --min-conf 0.6 --min-lift 1.5 --output "$OUT/mlxtend_basic_100k_s0p02_expected_fail/rules.csv"

"$PY" - "$OUT" <<'PYEND' | tee "$OUT/rss_summary.csv"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("label,status,max_rss_mib,elapsed_s")
for p in sorted(out.glob("*/metrics.json")):
    m = json.loads(p.read_text())
    print(f"{m['label']},{m['status']},{m['max_rss_kb']/1024:.1f},{m['elapsed_s']}")
PYEND
