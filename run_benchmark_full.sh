#!/bin/bash
#SBATCH --job-name=benchmark_full
#SBATCH --output=benchmark_full_%j.out
#SBATCH --error=benchmark_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G

# Scalability experiment: runs on the full wishlist_data.parquet dataset
# (Section 5.6 and Basic comparison in thesis). Only runs Experiments 2
# (Basic vs Cumulate) and the k-sweep at K=3, s=0.02 on the full dataset.

module load gcc/13.2.0
source /home/zrf539/myenv/bin/activate
cd /home/zrf539/BachelorBenchmark

python Benchmark.py Data/wishlist_data.parquet \
    --output-dir results/thesis_repro_full \
    --skip sensitivity example_rules k_sweep support_sweep \
           l0_pair_example rule_candidate_space held_out_recall \
    --basic-k 3 --basic-support 0.02 --basic-conf 0.6 --basic-lift 1.5 \
    --basic-max-len 5 \
    --repeats 3
