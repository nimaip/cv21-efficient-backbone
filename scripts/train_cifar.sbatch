#!/bin/bash
#SBATCH --job-name=p01-backbone
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hice1/nponna3/scratch/cv21/logs/%x_%j.out
#SBATCH --error=/home/hice1/nponna3/scratch/cv21/logs/%x_%j.err

# Run sbatch from /home/hice1/nponna3/scratch/cv21/p01-efficient-backbone
# Slurm executes a copy of this file, so the submit directory is the reliable path.
#   sbatch scripts/train_cifar.sbatch configs/cifar100_resnet18.yaml
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CV21_DATA="${CV21_DATA:-/home/hice1/nponna3/scratch/cv21/datasets}"
export CV21_CHECKPOINT_DIR="${CV21_CHECKPOINT_DIR:-/home/hice1/nponna3/scratch/cv21/checkpoints/p01}"
mkdir -p "$CV21_DATA" "$CV21_CHECKPOINT_DIR" /home/hice1/nponna3/scratch/cv21/logs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv21

python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "PyTorch is missing from the cv21 env. Install it before submitting this job."; exit 1; }

CONFIG="${1:-configs/cifar100_deit_tiny.yaml}"
echo "config=${CONFIG}"
if [[ -f .git_commit ]]; then
  echo "git_commit=$(cat .git_commit)"
fi
python -u -m efficient_backbone.train --config "$CONFIG"
