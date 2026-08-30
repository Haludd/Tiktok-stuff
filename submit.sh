#!/bin/bash
#SBATCH --job-name=tiktok-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
# If the default partition has no GPUs, uncomment and set this from `sinfo`:
##SBATCH --partition=CHANGEME

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
echo "workdir: $PWD"

# --- where to put the venv -------------------------------------------------
# Prefer scratch (usually unquotaed); fall back to the repo's parent.
if [ -d /scratch/work/athee258 ] && [ -w /scratch/work/athee258 ]; then
    BASE=/scratch/work/athee258
else
    BASE="$(cd .. && pwd)"
fi
VENV="$BASE/venv-tiktok"
export PIP_CACHE_DIR="$BASE/.pip-cache"
export TMPDIR="${TMPDIR:-$BASE/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
echo "venv: $VENV"

echo "=== gpu visible to this job ==="
nvidia-smi || { echo "FATAL: no GPU in this allocation"; exit 1; }

# --- build the environment once, reuse thereafter --------------------------
if [ ! -x "$VENV/bin/python" ]; then
    echo "=== creating venv ==="
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    # On Linux the default PyPI torch wheel is the CUDA build, so plain
    # requirements.txt is correct here (unlike on Windows).
    "$VENV/bin/pip" install -r requirements.txt
else
    echo "=== reusing existing venv ==="
fi

# --- refuse to waste the allocation on CPU ---------------------------------
echo "=== verifying CUDA ==="
"$VENV/bin/python" - <<'PY'
import sys, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: torch cannot see the GPU - aborting instead of training on CPU.")
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
PY

echo "=== training ==="
srun "$VENV/bin/python" train_new.py
echo "=== done at $(date) ==="
