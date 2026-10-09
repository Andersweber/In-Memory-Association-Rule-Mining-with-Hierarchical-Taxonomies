#!/bin/bash
#SBATCH --job-name=assoc_bench
#SBATCH --output=slurm_logs/assoc_bench_%j.out
#SBATCH --error=slurm_logs/assoc_bench_%j.err
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
cd ~/benchmark
module load anaconda3/2024.10-py3.12.7 2>/dev/null || true
module load gcc/13.2.0 2>/dev/null || true

# Record the node, CPU and compiler for the paper
echo "NODE: $(hostname)"
lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Thread'
g++ --version | head -1

# Rebuild the C++ binary for this node's CPU, the same way job 9442's binary was built
make -B -C apriori_cumulate/cpp CONDA_ENV=/opt/software/anaconda3/2024.10-py3.12.7 apriori_cumulate_cpp
ldd apriori_cumulate/cpp/apriori_cumulate_cpp | grep -E 'arrow|parquet|not found'
echo "BUILD OK"

# Run the full benchmark, same settings as job 9442
export PROFILE=full REPEATS=3 RUN_FULL=1
bash slurm/run_slurm_benchmark.sh
