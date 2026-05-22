#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=48G

module load gcc/13.2.0
source /home/zrf539/myenv/bin/activate
cd /home/zrf539/BachelorBenchmark

python Benchmark.py Data/samples/100000 \
    --output-dir results/thesis_repro \
    --catalogue-base Data/wishlist_data.parquet \
    --sweep-k 3 --sweep-support 0.02 \
    --sweep-tau 0.5 0.6 0.7 --sweep-lambda 1.2 1.5 2.0 \
    --sweep-max-ante-len 3 --sweep-max-cons-len 2 \
    --basic-k 3 --basic-support 0.02 --basic-conf 0.6 --basic-lift 1.5 \
    --basic-max-len 5 \
    --example-k 5 --example-support 0.02 --example-conf 0.6 --example-lift 1.5 \
    --example-max-ante-len 3 --example-max-cons-len 2 \
    --ksweep-k 1 2 3 4 5 \
    --ksweep-support 0.02 --ksweep-conf 0.6 --ksweep-lift 1.5 \
    --ksweep-max-ante-len 3 --ksweep-max-cons-len 2 \
    --ssweep-support 0.05 0.03 0.02 0.01 \
    --ssweep-k 3 --ssweep-conf 0.6 --ssweep-lift 1.5 \
    --ssweep-max-ante-len 3 --ssweep-max-cons-len 2 \
    --catalogue-k-levels 3 \
    --held-out-train-frac 0.8 \
    --held-out-mine-conf 0.50 \
    --held-out-final-conf 0.60 \
    --held-out-conf-sweep 0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
    --repeats 3
