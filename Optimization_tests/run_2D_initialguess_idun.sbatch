#!/bin/bash
#SBATCH --job-name=2D_initguess
#SBATCH --account=ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=1-7
#SBATCH --output=/cluster/work/pederava/idun_output/logs/2D_initguess_%A_%a.out
#SBATCH --error=/cluster/work/pederava/idun_output/logs/2D_initguess_%A_%a.err

set -euo pipefail

mkdir -p /cluster/work/pederava/idun_output/logs
mkdir -p /cluster/work/pederava/idun_output/initialguess

# Uncomment and adjust if your Idun login requires a Julia module:
# module purge
# module load Julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export IDUN_OUTPUT_DIR=/cluster/work/pederava/idun_output/initialguess

# Main tuning parameters
export LBFGS_MAX_ITERS=6
export RUN_GN_AFTER_LBFGS=false
export GN_AFTER_LBFGS_ITERS=1
export USE_FULL_GUESS_SET=true
export MAKE_PLOTS=true

echo "Starting array task ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Output directory: ${IDUN_OUTPUT_DIR}"
echo "Start time: $(date)"

julia --project=. 2D-opt_initialguess_idun.jl ${SLURM_ARRAY_TASK_ID}

echo "End time: $(date)"
