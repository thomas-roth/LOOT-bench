#!/usr/bin/env bash
# Launch a QC-FQL run against one g3a-bench task.
#
#   ./train_qc.sh <TASK_ID> <DATASET.npz> [extra main.py flags...]
#   ./train_qc.sh G3A-Stack3-Easy-Dk1_uni out/qc/stack.npz --offline_steps 100000
#
# Training lives in a separate checkout of upstream qc; this repository holds
# only the seam to it (docs/adr/002). The script exists for the settings that
# are invisible in the code and cost a whole run when they are wrong:
# preallocation, the venv the run is started from, the filesystem it writes to,
# and the wandb login. Each one is checked before the first simulator is built,
# because every one of them otherwise fails minutes to hours in.
#
# Environment overrides:
#   QC_DIR                the qc checkout          (default: ../qc)
#   QC_PYTHON             its interpreter          (default: $QC_DIR/.venv/bin/python)
#   RUN_DIR               where runs are written   (default: /var/tmp/$USER/qc-runs)
#   WANDB_MODE            online | offline | disabled  (default: online)
#   G3A_RUN_SCRIPT        names the launching script; becomes the script-<name>
#                         wandb tag                (default: train_qc)
#   G3A_START_POSE_NOISE  override task start-pose noise [rad] (default: task value)
#                         or pass --start_pose_noise
#   G3A_ALLOW_NO_TMUX     set to 1 to run outside tmux
#   CUDA_VISIBLE_DEVICES, and any other variable the run should inherit

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QC_DIR="${QC_DIR:-$(cd "$REPO/../qc" 2>/dev/null && pwd || echo "$REPO/../qc")}"
QC_PYTHON="${QC_PYTHON:-$QC_DIR/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-/var/tmp/${USER:-$(id -un)}/qc-runs}"
WANDB_MODE="${WANDB_MODE:-online}"

G3A_RUN_SCRIPT="${G3A_RUN_SCRIPT:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export G3A_RUN_SCRIPT

die() {
  echo "train_qc.sh: $1" >&2
  exit 1
}

usage() {
  # Header comment block, shebang excluded: self-adjusting, so it can't drift
  # from a hardcoded line range when the header above is edited.
  awk '/^#!/{next} /^#/{print; next} {exit}' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1
}

# --- arguments ---------------------------------------------------------------

[ $# -ge 2 ] || usage
TASK_ID="$1"
DATASET="$2"
shift 2

# Optional start-pose noise override (ADR-008). The task default is 0.0; a
# dataset recorded at 0.05 must run with the same value or
# g3a_bench.qc.dataset.check_dataset_matches_env aborts at load time.
# Accepts --start_pose_noise as a flag, or the G3A_START_POSE_NOISE env var.
# The value is passed to qc/main.py:main under --g3a_start_pose_noise.
G3A_START_POSE_NOISE="${G3A_START_POSE_NOISE:-}"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
  --start_pose_noise)
    [ $# -ge 2 ] || die "$1 needs a value (half-width [rad])"
    G3A_START_POSE_NOISE="$2"
    shift 2
    ;;
  --start_pose_noise=*)
    G3A_START_POSE_NOISE="${1#*=}"
    shift
    ;;
  --help | -h) usage ;;
  *)
    EXTRA_ARGS+=("$1")
    shift
    ;;
  esac
done
set -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
if [ -n "$G3A_START_POSE_NOISE" ]; then
  case "$G3A_START_POSE_NOISE" in
  '' | *[!0-9.\-+eE]*) die "G3A_START_POSE_NOISE='$G3A_START_POSE_NOISE' is not a number" ;;
  esac
  EXTRA_ARGS+=("--g3a_start_pose_noise=$G3A_START_POSE_NOISE")
  set -- "${EXTRA_ARGS[@]}"
fi

case "$TASK_ID" in
G3A-*) ;;
*) die "'$TASK_ID' is not a G3A task id; qc dispatches on the G3A- prefix" ;;
esac

[ -f "$DATASET" ] || die "no dataset at $DATASET (write one with g3a-convert-demos)"
DATASET="$(cd "$(dirname "$DATASET")" && pwd)/$(basename "$DATASET")"
case "$DATASET" in
*.npz) ;;
*) die "$DATASET is not an .npz" ;;
esac

[ -f "$QC_DIR/main.py" ] || die "no qc checkout at $QC_DIR (set QC_DIR)"
[ -x "$QC_PYTHON" ] || die "no interpreter at $QC_PYTHON (set QC_PYTHON)"

# --- wandb tags --------------------------------------------------------------
#
# A tag has to travel as a flag: wandb.init receives tags= explicitly, which
# beats the WANDB_TAGS environment variable.
#
# Only the dataset is tagged here, because only the shell knows it. main.py
# tags what its own flags say (`task-`, `script-`), so sending those too would
# tag the run twice with one fact. A caller with its own --wandb_tags
# (run_scripts/sweep.sh) owns the list and this build is skipped below.
#
# wandb caps a tag at 64 characters and rejects the run otherwise, and
# main.py splits --wandb_tags on `,` -- so a dataset stem is clamped and
# stripped of both before it reaches either, same as sweep.sh's own tags.
DATA_TAG="data-$(basename "$DATASET" .npz)"
DATA_TAG="${DATA_TAG//=/-}"
DATA_TAG="${DATA_TAG// /-}"
DATA_TAG="${DATA_TAG//,/-}"
TAGS=(--wandb_tags "${DATA_TAG:0:64}")
for arg in "$@"; do
  case "$arg" in
  --wandb_tags | --wandb_tags=*)
    TAGS=()
    break
    ;;
  esac
done

# --- the venv ----------------------------------------------------------------
#
# g3a-bench is reached by PYTHONPATH and must never be installed here: the
# install pulls a cu130 torch, and JAX then aborts at import. Both halves are
# checked, because an installed copy also shadows the checkout silently.

PYTHONPATH="$REPO:$QC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

# Without this JAX takes about 75 % of the card at import and mjlab's simulator
# has nothing left to allocate. Set before any qc code runs, not inside it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO="$REPO" "$QC_PYTHON" - <<'PY' || exit 1
import os, sys
from importlib import metadata, util

repo = os.environ["REPO"]
try:
    metadata.distribution("g3a-bench")
except metadata.PackageNotFoundError:
    pass
else:
    sys.exit(
        "train_qc.sh: g3a-bench is pip-installed in the qc venv. It drags in a "
        "cu130 torch that makes JAX abort at import. Uninstall it and let "
        "PYTHONPATH do the work."
    )

spec = util.find_spec("g3a_bench")
if spec is None or spec.origin is None:
    sys.exit(f"train_qc.sh: the qc venv cannot import g3a_bench from {repo}")
if not os.path.realpath(spec.origin).startswith(os.path.realpath(repo)):
    sys.exit(f"train_qc.sh: g3a_bench resolves to {spec.origin}, not {repo}")

import torch

cuda = torch.version.cuda or ""
if not cuda.startswith("12"):
    sys.exit(
        f"train_qc.sh: the qc venv has a CUDA-{cuda or '?'} torch. Keep it "
        "CUDA-12 only, or JAX aborts at import."
    )
print(f"[preflight] torch {torch.__version__}, g3a_bench from {repo}")
PY

# --- where the run writes ----------------------------------------------------
#
# $HOME here is NFSv4 with sec=krb5p and can refuse access in the middle of a
# run, which loses the run. So the check is the filesystem type, not the path:
# the repository itself sits on that mount.

mkdir -p "$RUN_DIR"
RUN_FS="$(stat -f -c %T "$RUN_DIR" 2>/dev/null || echo unknown)"
case "$RUN_FS" in
nfs*) die "$RUN_DIR is on $RUN_FS, which can EACCES mid-run. Set RUN_DIR to local disk." ;;
unknown) echo "[warn] cannot read the filesystem type of $RUN_DIR" >&2 ;;
esac

if [ -n "${UV_CACHE_DIR:-}" ] && [ "$(stat -f -c %T "$UV_CACHE_DIR" 2>/dev/null || echo unknown)" = "tmpfs" ]; then
  echo "[warn] UV_CACHE_DIR=$UV_CACHE_DIR is on a RAM disk; it fills and breaks installs" >&2
fi

# --- wandb -------------------------------------------------------------------
#
# wandb.init runs after the environment and the dataset are built, so an
# unauthenticated online run dies several minutes in.

export WANDB_MODE
if [ "$WANDB_MODE" = "online" ]; then
  "$QC_PYTHON" - <<'PY' || exit 1
import sys
import wandb

if not wandb.api.api_key:
    sys.exit(
        "train_qc.sh: no wandb login. Run `wandb login`, or launch with "
        "WANDB_MODE=offline."
    )
PY
fi

# --- tmux --------------------------------------------------------------------
#
# A run is hours long and the session is usually closed before it ends.

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "${G3A_ALLOW_NO_TMUX:-0}" != "1" ]; then
  die "not inside tmux; a closed terminal kills the run. Start 'tmux new', or set G3A_ALLOW_NO_TMUX=1."
fi

# --- launch ------------------------------------------------------------------

echo "[launch] task     $TASK_ID"
echo "[launch] dataset  $DATASET"
if [ -n "${G3A_START_POSE_NOISE:-}" ]; then
  echo "[launch] start_pose_noise $G3A_START_POSE_NOISE"
fi
echo "[launch] runs     $RUN_DIR  ($RUN_FS)"
echo "[launch] wandb    $WANDB_MODE"
echo "[launch] script   $G3A_RUN_SCRIPT"

cd "$QC_DIR"
set -x
exec "$QC_PYTHON" main.py \
  --env_name "$TASK_ID" \
  --g3a_dataset "$DATASET" \
  --save_dir "$RUN_DIR" \
  ${TAGS[@]+"${TAGS[@]}"} \
  "$@"
