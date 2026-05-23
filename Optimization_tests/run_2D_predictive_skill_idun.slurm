#!/bin/bash
#SBATCH --job-name=predskill
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-7
#SBATCH --output=/cluster/work/pederava/idun_output/predictive_skill/logs/predskill_%A_%a.out
#SBATCH --error=/cluster/work/pederava/idun_output/predictive_skill/logs/predskill_%A_%a.err

set -euo pipefail

# Change this if your code folder has another path on Idun.
cd "$HOME/Thesis Code/Optimization_tests"

mkdir -p /cluster/work/pederava/idun_output/predictive_skill/logs

export PREDICTIVE_SKILL_OUTPUT_DIR="/cluster/work/pederava/idun_output/predictive_skill"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export LBFGS_MAX_ITERS=6
export RUN_GN_AFTER_LBFGS=false
export GN_AFTER_LBFGS_ITERS=1
export MAKE_PLOTS=false
export SKIP_EXISTING=true

julia --project=. 2D_predictive_skill_idun.jl
