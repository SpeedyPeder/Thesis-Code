#!/bin/bash
#SBATCH --job-name=2D_init_sens
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-7
#SBATCH --output=/cluster/work/pederava/idun_output/logs/2D_init_sens_%A_%a.out
#SBATCH --error=/cluster/work/pederava/idun_output/logs/2D_init_sens_%A_%a.err

set -euo pipefail

mkdir -p /cluster/work/pederava/idun_output/logs
mkdir -p /cluster/work/pederava/idun_output/initialguess_sensitivity

# Uncomment and adjust if your Idun login requires a Julia module:
# module purge
# module load Julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export IDUN_OUTPUT_DIR=/cluster/work/pederava/idun_output/initialguess_sensitivity

# Main tuning parameters
export LBFGS_MAX_ITERS=6
export RUN_GN_AFTER_LBFGS=false
export GN_AFTER_LBFGS_ITERS=1
export MAKE_PLOTS=true

# Optional: can be useful if BLAS threading otherwise oversubscribes.
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "/cluster/home/pederava/master_thesis/Thesis Code"

echo "Starting sensitivity array task ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Output directory: ${IDUN_OUTPUT_DIR}"
echo "Start time: $(date)"

julia --project=. 2D-opt_initialguess_sensitivity_idun.jl ${SLURM_ARRAY_TASK_ID}

echo "End time: $(date)"
