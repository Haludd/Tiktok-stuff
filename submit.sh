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

# sbatch exports the submitting shell's env, so HOME may still point at the
# login node's /scratch, which compute nodes cannot write to. Repoint it, or
# torch.hub fails with EACCES when caching pretrained weights.
if [ ! -w "$HOME" ]; then
    echo "HOME ($HOME) not writable here; repointing to $BASE"
    export HOME="$BASE"
fi
export XDG_CACHE_HOME="$HOME/.cache"
export TORCH_HOME="$HOME/.cache/torch"
export PIP_CACHE_DIR="$BASE/.pip-cache"
export TMPDIR="${TMPDIR:-$BASE/tmp}"
export PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME"
echo "venv: $VENV   HOME: $HOME"

echo "=== gpu visible to this job ==="
nvidia-smi || { echo "FATAL: no GPU in this allocation"; exit 1; }

# --- build the environment once, reuse thereafter --------------------------
if [ ! -x "$VENV/bin/python" ]; then
    echo "=== creating venv ==="
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
fi

# The gpu-long nodes are TITAN V (Volta, compute capability 7.0). Default PyPI
# torch wheels are built against CUDA 13 and ship no sm_70 kernels, so they
# import fine and report cuda available, then fail on the first kernel launch.
# The cu126 build still includes sm_70. Bump the stamp to force a rebuild.
if [ ! -f "$VENV/.recipe-v2" ]; then
    echo "=== installing torch (cu126, has sm_70 kernels) ==="
    "$VENV/bin/pip" install --force-reinstall         --index-url https://download.pytorch.org/whl/cu126 torch torchvision
    echo "=== installing remaining requirements ==="
    "$VENV/bin/pip" install -r requirements.txt
    touch "$VENV/.recipe-v2"
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
print("capability:", torch.cuda.get_device_capability(0))
print("built for:", torch.cuda.get_arch_list())
# cuda.is_available() is not enough: a wheel without kernels for this GPU
# still reports True and only fails on the first real kernel launch.
try:
    x = torch.randn(64, 64, device="cuda")
    (x @ x).sum().item()
    torch.cuda.synchronize()
except Exception as e:
    print("FATAL: GPU present but unusable -", type(e).__name__, e)
    sys.exit(1)
print("kernel launch OK")
PY

echo "=== training ==="
srun "$VENV/bin/python" train_new.py
echo "=== done at $(date) ==="
