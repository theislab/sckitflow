#!/bin/bash
# Run a tahoe_train.py experiment as a standalone GPU batch job (one GPU node each), so several configs
# sweep in parallel. Submit e.g.:
#   sbatch -J lr1e3 scripts/run_tahoe_train.sbatch --lr 1e-3 --chunk-size 16 --n-train-steps 4000
# The plates glob is quoted so python (not the shell) expands it. Extra args ("$@") pass through.
#SBATCH -p gpu_p
#SBATCH -q gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
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
# Default plates glob + rep target the 2026-07-19 snakemake re-prep (plate*_filt_*.zarr,
# obsm/X_pca_pflog2048); pass --plates/--sample-rep again to override (argparse: last wins).
exec .venv/bin/python scripts/tahoe_train.py --plates "$DATA/plate*_filt_*.zarr" \
    --sample-rep X_pca_pflog2048 --device cuda "$@"
