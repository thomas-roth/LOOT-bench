#!/usr/bin/env bash
# Alpha x seed sweep of one g3a-bench task on QC-FQL. Why it is shaped this
# way: run_scripts/README.md.
#
#   sweep.sh <RUN_GROUP> <TASK_ID> <DATASET.npz> [options] [main.py flags...] [-- fixed flags...]
#
# Arguments:
#   RUN_GROUP        what this sweep is called; the flags typed on top of a
#                    wrapper are appended to it
#   TASK_ID          a G3A- task id
#   DATASET.npz      a relative path is read from the repository root
#
# Options:
#   --run-group NAME   replace RUN_GROUP
#   --num-seeds N      how many random seeds       (default: 3)
#   --seeds A,B,C      exact seeds, instead of random ones
#   --alphas A,B,C     the alphas to sweep         (default: 1000,10000)
#   --max-parallel N   runs going at once          (default: 3)
#   --tag TAG          an extra wandb tag; repeatable
#   --dry-run          print the runs that would start, start none
#
# Any other flag before `--` is passed to main.py, tagged, and appended to the
# group name; everything after `--` is passed to main.py only, which is where a
# wrapper puts its fixed configuration. Flags before `--` are placed after it on
# the command line, so they win.
#
# One sweep is one wandb group; the alphas in it are told apart by their
# alpha-<a> tag. The log goes into the folder main.py writes the run itself to:
#
#   $RUN_DIR/<task>/<run group>/alpha-<alpha>_seed-<seed>/train.log
#
# A failed run does not stop the others; the failures are named at the end and
# the exit status is non-zero. Ctrl-C stops the whole sweep.
#
# The repository and the dataset are read over Kerberized NFS here, so the
# sweep renews the Kerberos ticket for as long as it runs.
#
# Environment: RUN_DIR (default /var/tmp/$USER/qc-runs), G3A_RUN_SCRIPT (the
# script-<name> tag; a wrapper sets it), SWEEP_RENEW_INTERVAL (seconds between
# ticket renewals, default 1800; 0 turns the renewal off), plus what
# train_qc.sh reads.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-/var/tmp/${USER:-$(id -un)}/qc-runs}"
export RUN_DIR

G3A_RUN_SCRIPT="${G3A_RUN_SCRIPT:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export G3A_RUN_SCRIPT

DEFAULT_NUM_SEEDS=3
DEFAULT_ALPHAS="1000,10000"
DEFAULT_MAX_PARALLEL=3

die() {
  echo "sweep.sh: $1" >&2
  exit 1
}

usage() {
  sed -n '2,41p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit "${1:-1}" # asked for it: 0; got the arguments wrong: 1
}

# A tag and a group name are both read by a human and used as a path
# component, so neither may carry `=`, `/` or a space. Case is kept:
# `sparse-True` is the flag as it was typed, and lowercasing it only makes the
# tag harder to match against the command that produced it.
sanitize() {
  local s="$1"
  s="${s#--}"
  s="${s//=/-}"
  s="${s//\//-}"
  s="${s// /-}"
  s="${s//,/-}"
  printf '%s' "$s"
}

# wandb caps a tag at 64 characters and rejects the run otherwise -- minutes
# into the run, once the environment and the dataset are built. Clamping the
# assembled tag rather than its parts is the point: `data-` plus a dataset stem
# is already 59 characters here, and a key-value pair concatenates two of them.
TAG_MAX=64
clamp() {
  printf '%s' "${1:0:$TAG_MAX}"
}

# Whether the next token is a value rather than the next flag. A negative
# number is a value: `--save_interval -1` is one flag, not two.
is_value() {
  case "${1-}" in
  "") return 0 ;;
  -[0-9]*) return 0 ;;
  -*) return 1 ;;
  *) return 0 ;;
  esac
}

# --- arguments ---------------------------------------------------------------

# Before the count, or `sweep.sh --help` is short of positionals and answered
# with the same usage text but a failure status.
case "${1-}" in -h | --help) usage 0 ;; esac

[ $# -ge 3 ] || usage
RUN_GROUP="$1"
TASK_ID="$2"
DATASET="$3"
shift 3

NUM_SEEDS="$DEFAULT_NUM_SEEDS"
SEEDS=""
ALPHAS="$DEFAULT_ALPHAS"
MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
DRY_RUN=0
EXTRA_TAGS=()
USER_FLAGS=() # typed on top of the wrapper: tagged and named
BASE_FLAGS=() # the wrapper's own configuration: neither
NAME_PARTS=()
FLAG_TAGS=()

# A flag typed on top of the wrapper identifies the run: it goes into the tags
# so the run list can be filtered by it, and into the group name so two
# settings of one wrapper are two groups. Only flags that carry a value do --
# a bare `--sparse` has nothing to name.
note_flag() {
  local part
  part="$(sanitize "$1")-$(sanitize "$2")"
  NAME_PARTS+=("$part")
  FLAG_TAGS+=("$(clamp "$part")")
}

while [ $# -gt 0 ]; do
  case "$1" in
  --)
    shift
    BASE_FLAGS=("$@")
    break
    ;;
  -h | --help) usage 0 ;;
  --dry-run)
    DRY_RUN=1
    shift
    ;;
  --run-group)
    is_value "${2-}" || die "--run-group needs a name"
    RUN_GROUP="$2"
    shift 2
    ;;
  --run-group=*)
    RUN_GROUP="${1#*=}"
    shift
    ;;
  --num-seeds)
    [ $# -ge 2 ] || die "--num-seeds needs a number"
    NUM_SEEDS="$2"
    shift 2
    ;;
  --num-seeds=*)
    NUM_SEEDS="${1#*=}"
    shift
    ;;
  --seeds)
    [ $# -ge 2 ] || die "--seeds needs a comma-separated list"
    SEEDS="$2"
    shift 2
    ;;
  --seeds=*)
    SEEDS="${1#*=}"
    shift
    ;;
  --alphas)
    [ $# -ge 2 ] || die "--alphas needs a comma-separated list"
    ALPHAS="$2"
    shift 2
    ;;
  --alphas=*)
    ALPHAS="${1#*=}"
    shift
    ;;
  --max-parallel)
    [ $# -ge 2 ] || die "--max-parallel needs a number"
    MAX_PARALLEL="$2"
    shift 2
    ;;
  --max-parallel=*)
    MAX_PARALLEL="${1#*=}"
    shift
    ;;
  --tag)
    is_value "${2-}" || die "--tag needs a tag"
    EXTRA_TAGS+=("$(clamp "$(sanitize "$2")")")
    shift 2
    ;;
  --tag=*)
    EXTRA_TAGS+=("$(clamp "$(sanitize "${1#*=}")")")
    shift
    ;;
  --*=*)
    USER_FLAGS+=("$1")
    note_flag "${1%%=*}" "${1#*=}"
    shift
    ;;
  --*)
    if [ $# -ge 2 ] && is_value "$2"; then
      USER_FLAGS+=("$1" "$2")
      note_flag "$1" "$2"
      shift 2
    else
      USER_FLAGS+=("$1")
      shift
    fi
    ;;
  *) die "unexpected argument '$1'; main.py flags start with --" ;;
  esac
done

case "$TASK_ID" in
G3A-*) ;;
*) die "'$TASK_ID' is not a G3A task id" ;;
esac

case "$DATASET" in /*) ;; *) DATASET="$REPO/$DATASET" ;; esac
# Checked here rather than once per run later: train_qc.sh builds a simulator
# before it ever opens the dataset, so a wrong path costs the whole start-up
# before it says so.
[ -f "$DATASET" ] || die "dataset not found: $DATASET"

[ -n "$RUN_GROUP" ] || die "the run group name is empty"
RUN_GROUP="$(sanitize "$RUN_GROUP")"

case "$NUM_SEEDS" in '' | *[!0-9]*) die "--num-seeds wants a number, got '$NUM_SEEDS'" ;; esac
case "$MAX_PARALLEL" in '' | *[!0-9]*) die "--max-parallel wants a number, got '$MAX_PARALLEL'" ;; esac
[ "$MAX_PARALLEL" -ge 1 ] || die "--max-parallel must be at least 1"
[ -n "$ALPHAS" ] || die "--alphas is empty"

# --- the sweep grid ----------------------------------------------------------

alphas=()
IFS=',' read -r -a alphas <<<"$ALPHAS"
# main.py names the run folder from the same alpha and, being handed a float,
# renders `1000.0` as `1000`. Spelling it that way here too is what keeps
# train.log in the run's folder rather than in an empty one beside it. What is
# not a decimal number is refused here rather than by absl, minutes in.
normalized=()
for a in "${alphas[@]}"; do
  [ -n "$a" ] || die "--alphas has an empty entry: '$ALPHAS'"
  case "$a" in
  *[!0-9.]* | *.*.* | .* | *.) die "--alphas wants plain decimal numbers, got '$a'" ;;
  esac
  case "$a" in
  *.*)
    while [ "${a%0}" != "$a" ]; do a="${a%0}"; done # "0.50" -> "0.5", "1000.0" -> "1000."
    a="${a%.}"
    ;;
  esac
  normalized+=("$a")
done
alphas=("${normalized[@]}")
# The alphas- tag then names what ran, not what was typed.
ALPHAS="$(
  IFS=,
  echo "${alphas[*]}"
)"

# One set of seeds for every alpha, not a fresh draw per alpha: the alphas are
# then compared on the same seeds, and a difference between them is not a
# difference between seed draws.
if [ -n "$SEEDS" ]; then
  seeds=()
  IFS=',' read -r -a seeds <<<"$SEEDS"
  for s in "${seeds[@]}"; do
    case "$s" in '' | *[!0-9]*) die "--seeds wants numbers, got '$SEEDS'" ;; esac
  done
  NUM_SEEDS="${#seeds[@]}"
else
  [ "$NUM_SEEDS" -ge 1 ] || die "--num-seeds must be at least 1"
  seeds=()
  while [ "${#seeds[@]}" -lt "$NUM_SEEDS" ]; do
    s=$((RANDOM * 32768 + RANDOM))
    case " ${seeds[*]-} " in *" $s "*) continue ;; esac # Resample if drawn twice.
    seeds+=("$s")
  done
fi

# The group name carries what was typed on top of the wrapper. The sweep's own
# settings do not go in it -- --num-seeds does not say what the experiment is --
# but they do become tags when they are not the default.
if [ "${#NAME_PARTS[@]}" -gt 0 ]; then
  RUN_GROUP="$RUN_GROUP$(printf '_%s' "${NAME_PARTS[@]}")"
fi
# setup_wandb tags every run with its group, so the group name is under the
# same cap as the tags. Truncating it silently would make two sweeps share a
# name, which is exactly what the name is for.
if [ "${#RUN_GROUP}" -gt "$TAG_MAX" ]; then
  RUN_GROUP="$(clamp "$RUN_GROUP")"
  echo "[sweep] WARNING: group name over $TAG_MAX chars, cut to '$RUN_GROUP'." >&2
  echo "[sweep] WARNING: pass a shorter --run-group if that collides." >&2
fi

# No `task-`/`script-` tags: main.py has --env_name and $G3A_RUN_SCRIPT and tags
# both itself.
SWEEP_TAGS=(
  "$(clamp "data-$(sanitize "$(basename "$DATASET" .npz)")")"
)
if [ -z "$SEEDS" ] && [ "$NUM_SEEDS" != "$DEFAULT_NUM_SEEDS" ]; then
  SWEEP_TAGS+=("num-seeds-$NUM_SEEDS")
fi
if [ "$MAX_PARALLEL" != "$DEFAULT_MAX_PARALLEL" ]; then
  SWEEP_TAGS+=("max-parallel-$MAX_PARALLEL")
fi
if [ "$ALPHAS" != "$DEFAULT_ALPHAS" ]; then
  SWEEP_TAGS+=("$(clamp "alphas-$(sanitize "$ALPHAS")")")
fi
SWEEP_TAGS+=(${FLAG_TAGS[@]+"${FLAG_TAGS[@]}"})
SWEEP_TAGS+=(${EXTRA_TAGS[@]+"${EXTRA_TAGS[@]}"})

GROUP_DIR="$RUN_DIR/$TASK_ID/$RUN_GROUP"

echo "[sweep] group    $RUN_GROUP"
echo "[sweep] task     $TASK_ID"
echo "[sweep] data     $DATASET"
echo "[sweep] alphas   ${alphas[*]}"
echo "[sweep] seeds    ${seeds[*]}"
echo "[sweep] parallel $MAX_PARALLEL of $((${#alphas[@]} * ${#seeds[@]})) runs"
echo "[sweep] tags     $(
  IFS=,
  echo "${SWEEP_TAGS[*]}"
)"
echo "[sweep] runs     $GROUP_DIR"

# The group name has no timestamp in it, so the same sweep launched twice keeps
# the same name and the runs accumulate in one wandb group. That is sometimes
# what is wanted (three more seeds on the same question) and sometimes two
# sessions that will be hard to tell apart, so it warns and continues.
if [ -d "$GROUP_DIR" ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "[sweep] WARNING: $GROUP_DIR exists; these runs join that wandb group." >&2
  echo "[sweep] WARNING: pass --run-group NAME for a separate one. Continuing in 3s." >&2
  sleep 3
fi

# --- the kerberos ticket -----------------------------------------------------

# $REPO, the venvs and the dataset are read over NFS mounted with sec=krb5p, and
# a sweep outlives a login ticket. When the ticket expires the runs do not fail
# gracefully: they die with SIGBUS (the .venv pages they mmap'd stop being
# readable) or status 126 (train_qc.sh can no longer be executed). The renewer
# is a child of the sweep, so it leaves nothing behind afterwards.
RENEW_INTERVAL="${SWEEP_RENEW_INTERVAL:-1800}"
RENEW_PID=""
PAM_CCACHE="/tmp/krb5cc_$(id -u)"

# Read from a local disk there is nothing to renew.
repo_needs_ticket() {
  local opts
  opts="$(findmnt -no OPTIONS --target "$REPO" 2>/dev/null)" || return 1
  case "$opts" in
  *sec=krb5*) return 0 ;;
  *) return 1 ;;
  esac
}

start_renewer() {
  [ "$RENEW_INTERVAL" != 0 ] || return 0
  repo_needs_ticket || return 0
  command -v kinit >/dev/null 2>&1 || die "$REPO needs a Kerberos ticket, but kinit is missing"
  # Refused here rather than hours in, when the first run touches the share.
  klist -s 2>/dev/null || die "no valid Kerberos ticket for $REPO; run kinit first"
  klist 2>/dev/null | grep -q 'renew until' ||
    echo "[sweep] WARNING: the ticket cannot be renewed; kinit -r 14d before a long sweep." >&2
  # PAM deletes its own cache when any session of this user logs out, including
  # one that has nothing to do with the sweep; renewing does not bring it back.
  case "${KRB5CCNAME:-FILE:$PAM_CCACHE}" in
  *"$PAM_CCACHE")
    echo "[sweep] WARNING: the ticket cache is PAM's $PAM_CCACHE, which is deleted" >&2
    echo "[sweep] WARNING: when any session of yours logs out. Copy it to local disk and" >&2
    echo "[sweep] WARNING: point KRB5CCNAME at the copy before a long sweep." >&2
    ;;
  esac
  (
    # The trap takes the sleep along, so a stopped sweep leaves none behind.
    trap 'kill $! 2>/dev/null; exit 0' TERM
    while :; do
      sleep "$RENEW_INTERVAL" &
      wait $! 2>/dev/null || true
      kinit -R 2>/dev/null || {
        echo "[sweep] WARNING: kinit -R failed; the runs will die when the ticket expires" >&2
        exit 1
      }
    done
  ) &
  RENEW_PID="$!"
  echo "[sweep] ticket   renewed every ${RENEW_INTERVAL}s by pid $RENEW_PID"
}

# Both traps call this before stop_rest, whose bare `wait` would otherwise wait
# for a renewer that never returns.
stop_renewer() {
  [ -n "$RENEW_PID" ] || return 0
  kill "$RENEW_PID" 2>/dev/null || true
  wait "$RENEW_PID" 2>/dev/null || true
  RENEW_PID=""
}

# --- launch ------------------------------------------------------------------

start_run() {
  local alpha="$1" seed="$2" log_dir tags
  log_dir="$GROUP_DIR/alpha-${alpha}_seed-$seed"
  tags="$(
    IFS=,
    echo "${SWEEP_TAGS[*]},alpha-$alpha,seed-$seed"
  )"

  # The wrapper's flags come first and the flags typed on top of it second, so
  # a flag given on the command line beats the wrapper's default (absl takes
  # the last occurrence). The sweep's own flags come last: --seed, the alpha
  # and the group are what the sweep *is*, and are not overridable from either.
  set -- "$TASK_ID" "$DATASET" \
    ${BASE_FLAGS[@]+"${BASE_FLAGS[@]}"} \
    ${USER_FLAGS[@]+"${USER_FLAGS[@]}"} \
    --seed="$seed" \
    --agent.alpha="$alpha" \
    --run_group="$RUN_GROUP" \
    --wandb_tags="$tags"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] alpha $alpha seed $seed -> $log_dir/train.log"
    printf '[dry-run]   %s/train_qc.sh' "$REPO"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  mkdir -p "$log_dir"
  "$REPO/train_qc.sh" "$@" >"$log_dir/train.log" 2>&1 &
  pids+=("$!")
  cell_of["$!"]="alpha-$alpha seed-$seed"
  echo "[sweep] alpha $alpha seed $seed started, pid $!, log $log_dir/train.log"
}

# Alpha-major, so the first alpha is finished first if the sweep is cut short.
queue=()
for alpha in "${alphas[@]}"; do
  for seed in "${seeds[@]}"; do
    queue+=("$alpha $seed")
  done
done

if [ "$DRY_RUN" -eq 1 ]; then
  for cell in "${queue[@]}"; do
    # shellcheck disable=SC2086  # a queue entry is "<alpha> <seed>", split on purpose
    start_run $cell
  done
  echo "[dry-run] ${#queue[@]} run(s) not started"
  exit 0
fi

pids=()
declare -A cell_of=()
running=()
failed=()

# Ctrl-C stops every run of the sweep.
stop_rest() {
  for pid in ${running[@]+"${running[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'echo "[sweep] interrupted, stopping the runs" >&2; stop_renewer; stop_rest; exit 130' INT TERM
trap stop_renewer EXIT

start_renewer

reap_one() {
  local status=0 done_pid="" rest=()
  wait -n -p done_pid ${running[@]+"${running[@]}"} || status=$?
  local cell="${cell_of[${done_pid:-}]:-?}"
  for pid in ${running[@]+"${running[@]}"}; do
    [ "$pid" = "${done_pid:-}" ] || rest+=("$pid")
  done
  running=(${rest[@]+"${rest[@]}"})
  if [ "$status" -ne 0 ]; then
    echo "[sweep] $cell failed with status $status, see $GROUP_DIR/${cell// /_}/train.log" >&2
    failed+=("$cell")
  else
    echo "[sweep] $cell done"
  fi
  echo "[sweep] ${#running[@]} run(s) going, $((${#queue[@]} - next)) waiting"
}

next=0
while [ "$next" -lt "${#queue[@]}" ] || [ "${#running[@]}" -gt 0 ]; do
  while [ "$next" -lt "${#queue[@]}" ] && [ "${#running[@]}" -lt "$MAX_PARALLEL" ]; do
    # shellcheck disable=SC2086  # a queue entry is "<alpha> <seed>", split on purpose
    start_run ${queue[$next]}
    running+=("${pids[-1]}")
    next=$((next + 1))
  done
  reap_one
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[sweep] group $RUN_GROUP: failed: ${failed[*]}" >&2
  exit 1
fi
echo "[sweep] group $RUN_GROUP: all runs done"
