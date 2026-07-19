#!/bin/bash
# Run a tahoe_train.py experiment as a standalone GPU batch job (one GPU node each), so several configs
# sweep in parallel. Submit e.g.:
#   sbatch -J lr1e3 scripts/run_tahoe_train.sbatch --lr 1e-3 --chunk-size 16 --n-train-steps 4000
# The plates glob is quoted so python (not the shell) expands it. Extra args ("$@") pass through.
#SBATCH -p gpu_p
#SBATCH -q gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 02:00:00
#SBATCH --nice=5000
#SBATCH -o /lustre/groups/ml01/workspace/selman.ozleyen/projects/sc-flow-tools/runs/%x_%j.log

set -euo pipefail
cd /lustre/groups/ml01/workspace/selman.ozleyen/projects/sc-flow-tools
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DATA=/lustre/groups/ml01/datasets/selman.ozleyen/tahoe100_converted
echo "### $(date) job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
exec .venv/bin/python scripts/tahoe_train.py --plates "$DATA/plate*.zarr" --device cuda "$@"
