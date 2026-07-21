#!/bin/bash
#SBATCH -p gpu_p
#SBATCH -q gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH --nice=5000
#SBATCH -o /lustre/groups/ml01/workspace/selman.ozleyen/projects/sc-flow-tools/runs/%x_%j.log
set -euo pipefail
cd /lustre/groups/ml01/workspace/selman.ozleyen/projects/sc-flow-tools
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DATA=/lustre/groups/ml01/datasets/selman.ozleyen/tahoe100_converted
echo "### $(date) job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname)"
# plate*_filt_*.zarr + X_pca_pflog2048 = the 2026-07-19 snakemake re-prep; pass --plates/--sample-rep
# again to eval a different rep (e.g. raw X_pca, if it's ever regenerated).
exec .venv/bin/python scripts/tahoe_eval.py --plates "$DATA/plate*_filt_*.zarr" \
     --eval-plate "$DATA/plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.zarr" \
     --sample-rep X_pca_pflog2048 --device cuda "$@"
