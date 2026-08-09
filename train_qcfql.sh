#!/usr/bin/env bash
# Launch a QC-FQL run (upstream `qc`) so that a hiccup on the NFS home cannot
# take it down mid-training.
#
# The home directory is NFSv4 with sec=krb5p: every access needs a valid GSS
# credential, and a momentary rejection surfaces as EACCES on an ordinary
# write. A run that logs to the home on every step is therefore only as
# reliable as that credential. This script keeps all live I/O -- checkpoints,
# CSVs, wandb -- on local disk and mirrors the results back periodically, so a
# failed mirror costs one sync instead of the whole run.
#
# Usage:
#   ./train_qcfql.sh --env_name=G3A-Place-Easy-Dk1_uni \
#       --g3a_dataset_path=../g3a-bench/dataset/demos/.../demos.npz \
#       --offline_steps=1000000 --online_steps=1000000
#
# All arguments are forwarded to qc's main.py verbatim; --save_dir is injected.
#
# Environment overrides:
#   QC_DIR         path to the qc checkout      (default: ../qc)
#   QC_RUN_ROOT    local scratch root           (default: /tmp/$USER/qc-runs)
#   QC_MIRROR_DIR  where results are mirrored   (default: $QC_DIR/runs)
#   QC_MIRROR_EVERY  seconds between mirrors    (default: 600)

set -euo pipefail

G3A_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QC_DIR="${QC_DIR:-$(cd "$G3A_DIR/../qc" 2>/dev/null && pwd || true)}"
QC_RUN_ROOT="${QC_RUN_ROOT:-/tmp/$USER/qc-runs}"
QC_MIRROR_EVERY="${QC_MIRROR_EVERY:-600}"

die() { printf 'error: %s\n' "$1" >&2; exit 1; }
note() { printf '[train_qcfql] %s\n' "$1"; }

# --- preflight -------------------------------------------------------------

[ -n "$QC_DIR" ] && [ -f "$QC_DIR/main.py" ] || die "qc checkout not found (QC_DIR=${QC_DIR:-unset})"
[ -x "$QC_DIR/.venv/bin/python" ] || die "qc venv missing: $QC_DIR/.venv/bin/python"
[ $# -gt 0 ] || die "no arguments; pass main.py flags, e.g. --env_name=G3A-Place-Easy-Dk1_uni"

for arg in "$@"; do
    case "$arg" in
        --save_dir=*|--save_dir) die "--save_dir is managed by this script (see QC_RUN_ROOT)" ;;
    esac
done

# A Kerberos ticket is what makes NFS access yours rather than borrowed from
# the host's machine credential. Without one you are exposed to whatever
# happens to that shared credential -- which is what killed the run of
# 2026-08-08. Not fatal here, because the mirror is the only thing that needs
# the home mount, but worth knowing before starting a multi-hour job.
if command -v klist >/dev/null 2>&1 && ! klist -s 2>/dev/null; then
    note "WARNING: no Kerberos ticket (klist -s failed). Run 'kinit' so the"
    note "         result mirror to the NFS home cannot be denied. Continuing."
fi

# Extract env_name for a readable run directory; fall back to a timestamp.
env_name="run"
for arg in "$@"; do
    case "$arg" in
        --env_name=*) env_name="${arg#--env_name=}" ;;
    esac
done

stamp="$(date +%Y%m%d_%H%M%S)"
run_name="${env_name}_${stamp}"
RUN_DIR="$QC_RUN_ROOT/$run_name"
MIRROR_TARGET="${QC_MIRROR_DIR:-$QC_DIR/runs}/$run_name"

mkdir -p "$RUN_DIR/exp"

avail_kb="$(df -Pk "$RUN_DIR" | awk 'NR==2 {print $4}')"
[ "$avail_kb" -gt 10485760 ] || die "less than 10 GB free at $RUN_DIR (checkpoints are ~60 MB each)"

# --- mirror ----------------------------------------------------------------

# Mirrors the whole run dir (checkpoints, CSVs, wandb, log) as one
# self-contained tree. Failure is never fatal: the authoritative copy is the
# local one, and the next sync -- or the final one on exit -- catches up.
mirror() {
    mkdir -p "$MIRROR_TARGET" 2>/dev/null || { note "WARNING: cannot create $MIRROR_TARGET"; return 0; }
    if rsync -a --delete "$RUN_DIR/" "$MIRROR_TARGET/" 2>/dev/null; then
        note "mirrored -> $MIRROR_TARGET"
    else
        note "WARNING: mirror to $MIRROR_TARGET failed (results remain in $RUN_DIR)"
    fi
}

mirror_loop() {
    while sleep "$QC_MIRROR_EVERY"; do mirror; done
}

cleanup() {
    local rc=$?
    [ -n "${mirror_pid:-}" ] && kill "$mirror_pid" 2>/dev/null || true
    note "run finished (exit $rc); final mirror"
    mirror
    note "local run dir: $RUN_DIR"
    note "log:           $RUN_DIR/train.log"
}
trap cleanup EXIT

# --- run -------------------------------------------------------------------

note "run dir:  $RUN_DIR"
note "mirror:   $MIRROR_TARGET (every ${QC_MIRROR_EVERY}s)"
note "wandb:    offline"

mirror_loop &
mirror_pid=$!

cd "$QC_DIR"
WANDB_MODE=offline \
WANDB_DIR="$RUN_DIR" \
PYTHONPATH="$G3A_DIR:${PYTHONPATH:-}" \
    "$QC_DIR/.venv/bin/python" main.py "$@" --save_dir="$RUN_DIR/exp" 2>&1 \
    | tee "$RUN_DIR/train.log"

exit "${PIPESTATUS[0]}"
