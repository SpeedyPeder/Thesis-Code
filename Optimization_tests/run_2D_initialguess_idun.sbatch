#!/bin/bash
#SBATCH --job-name=2D_initguess
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-7%1
#SBATCH --output=/cluster/work/pederava/idun_output/logs/2D_initguess_%A_%a.out
#SBATCH --error=/cluster/work/pederava/idun_output/logs/2D_initguess_%A_%a.err

set -euo pipefail

mkdir -p /cluster/work/pederava/idun_output/logs
mkdir -p /cluster/work/pederava/idun_output/initialguess

# Uncomment and adjust if your Idun login requires a Julia module:
# module purge
# module load Julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export IDUN_OUTPUT_DIR=/cluster/work/pederava/idun_output/initialguess

# Main tuning parameters
export LBFGS_MAX_ITERS=6
export RUN_GN_AFTER_LBFGS=false
export GN_AFTER_LBFGS_ITERS=1
export USE_FULL_GUESS_SET=true
export MAKE_PLOTS=true

cd "/cluster/home/pederava/master_thesis/Thesis Code/Optimization_tests"

echo "Starting array task ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Output directory: ${IDUN_OUTPUT_DIR}"
echo "Start time: $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "SLURM array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Julia threads: ${JULIA_NUM_THREADS}"

julia --project=. 2D-opt_initialguess_idun.jl ${SLURM_ARRAY_TASK_ID}

echo "End time: $(date)"
