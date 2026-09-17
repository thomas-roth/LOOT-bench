#!/usr/bin/env bash
# Alpha x seed sweep of one loot-bench task on QC-FQL. Why it is shaped this
# way: run_scripts/README.md.
#
#   sweep.sh <RUN_GROUP> <TASK_ID> <DATASET.npz> [options] [main.py flags...] [-- fixed flags...]
#
# Arguments:
#   RUN_GROUP        what this sweep is called; the flags typed on top of a
#                    wrapper are appended to it
#   TASK_ID          a LOOT- task id
#   DATASET.npz      a relative path is read from the repository root
#
# Options:
#   --run-group NAME   replace RUN_GROUP
#   --num-seeds N      how many random seeds       (default: 3)
#   --seeds A,B,C      exact seeds, instead of random ones
#   --alphas A,B,C     the alphas to sweep         (default: 1,10,100)
#   --max-parallel N   runs going at once          (default: 3)
#   --tag TAG          an extra wandb tag; repeatable
#   --resume           continue this group: skip cells with token.tk, restart
#                      the rest from their newest params_<step>.pkl (cells with
#                      no checkpoint start fresh). Grid and flags are checked
#                      against GROUP_DIR/sweep.json; --num-seeds is refused and
#                      --seeds/--alphas must be a subset of the stored grid.
#   --overwrite        delete GROUP_DIR and start over (refused together with
#                      --resume). Without either flag, an existing GROUP_DIR
#                      aborts the launch instead of overwriting it.
#   --dry-run          print the runs that would start, start none
#
# Resuming is a same-host operation: RUN_DIR defaults to node-local /var/tmp
# (and train_qc.sh refuses shared filesystems), so another host sees no
# checkpoints. From a scheduler without tmux, export LOOT_ALLOW_NO_TMUX=1.
# Adding seeds to a finished question is a new --run-group, not a relaunch:
# --resume never expands the stored grid.
#
# Any other flag before `--` is passed to main.py, tagged, and appended to the
# group name; everything after `--` is passed to main.py only, which is where a
# wrapper puts its fixed configuration. Flags before `--` are placed after it on
# the command line, so they win.
#
# Every run trains QC-FQL, evaluated over 50 episodes with 6 recorded as
# video, unless a wrapper or flag overrides it.
#
# One sweep is one wandb group; the alphas in it are told apart by their
# alpha-<a> tag. The log goes into the folder main.py writes the run itself to:
#
#   $RUN_DIR/<task>/<run group>/alpha-<alpha>_seed-<seed>/train.log
#
# A failed run does not stop the others; the failures are named at the end and
# the exit status is non-zero. Ctrl-C stops the whole sweep.
#
# The repository, the venvs and the dataset are read over Kerberized NFS here,
# so the sweep refuses to start without a renewable ticket (kinit -r 14d) and
# renews it from a detached watcher logging to <group>/krb-renew.log. A run
# dying with status 126/135 means the ticket died, not the experiment: one such
# death costs a 60 s probe and a relaunch (a seconds-long NFS blip looks the
# same), and only a second one stops the queue instead of launching the rest
# into the same wall.
#
# Environment: RUN_DIR (default /var/tmp/$USER/qc-runs), LOOT_RUN_SCRIPT (the
# script-<name> tag; a wrapper sets it), SWEEP_RENEW_INTERVAL (seconds between
# ticket renewals, default 1800; 0 turns the renewal off against advice), plus
# what train_qc.sh reads.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-/var/tmp/${USER:-$(id -un)}/qc-runs}"
export RUN_DIR

LOOT_RUN_SCRIPT="${LOOT_RUN_SCRIPT:-$(basename "${BASH_SOURCE[0]}" .sh)}"
export LOOT_RUN_SCRIPT

DEFAULT_NUM_SEEDS=3
DEFAULT_ALPHAS="1,10,100"
DEFAULT_MAX_PARALLEL=3

die() {
  echo "sweep.sh: $1" >&2
  exit 1
}

usage() {
  # Header comment block, shebang excluded: self-adjusting, so it can't drift
  # from a hardcoded line range when the header above is edited.
  awk '/^#!/{next} /^#/{print; next} {exit}' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
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
SEEDS_GIVEN=0
ALPHAS="$DEFAULT_ALPHAS"
ALPHAS_GIVEN=0
MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
DRY_RUN=0
RESUME=0
OVERWRITE=0
NUM_SEEDS_GIVEN=0
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
  --resume)
    RESUME=1
    shift
    ;;
  --overwrite)
    OVERWRITE=1
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
    is_value "${2-}" || die "--num-seeds needs a number"
    NUM_SEEDS="$2"
    NUM_SEEDS_GIVEN=1
    shift 2
    ;;
  --num-seeds=*)
    NUM_SEEDS="${1#*=}"
    NUM_SEEDS_GIVEN=1
    shift
    ;;
  --seeds)
    is_value "${2-}" || die "--seeds needs a comma-separated list"
    SEEDS="$2"
    SEEDS_GIVEN=1
    shift 2
    ;;
  --seeds=*)
    SEEDS="${1#*=}"
    SEEDS_GIVEN=1
    shift
    ;;
  --alphas)
    is_value "${2-}" || die "--alphas needs a comma-separated list"
    ALPHAS="$2"
    ALPHAS_GIVEN=1
    shift 2
    ;;
  --alphas=*)
    ALPHAS="${1#*=}"
    ALPHAS_GIVEN=1
    shift
    ;;
  --max-parallel)
    is_value "${2-}" || die "--max-parallel needs a number"
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
LOOT-*) ;;
*) die "'$TASK_ID' is not a LOOT task id" ;;
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

# No `task-`/`script-` tags: main.py has --env_name and $LOOT_RUN_SCRIPT and tags
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
SWEEP_JSON="$GROUP_DIR/sweep.json"

# main.py flags this sweep owns: defaults first, then the wrapper's flags,
# then the flags typed on top, so each stage overrides the one before (absl
# takes the last occurrence). --save_interval lives here, not in main.py, so
# direct train_qc.sh runs keep main.py's default (-1, off). 50k is ~1 s and
# 54 MB per save against hours of training, and divides the 100k/300k/400k
# step counts the wrappers use (not every eval_interval out there: 20k/30k
# evals simply fall between checkpoints).
DEFAULT_MAIN_FLAGS=(
  --agent=agents/acfql.py
  --agent.normalize_q_loss=True
  --eval_episodes 50
  --video_episodes 6
  --save_interval 50000
)
EFF_ARGV=(
  "${DEFAULT_MAIN_FLAGS[@]}"
  ${BASE_FLAGS[@]+"${BASE_FLAGS[@]}"}
  ${USER_FLAGS[@]+"${USER_FLAGS[@]}"}
)

DS_MTIME="$(stat -c %Y "$DATASET")"
DS_SIZE="$(stat -c %s "$DATASET")"
QC_SHA=""
if [ -f "$REPO/../qc/main.py" ]; then
  QC_SHA="$(git -C "$REPO/../qc" rev-parse HEAD 2>/dev/null || true)"
fi

# --- resume / overwrite ------------------------------------------------------
#
# A relaunch cannot tell "continue my dead sweep" from "new experiment, same
# name", so continuing is an explicit --resume and replacing an explicit
# --overwrite; anything else aborts instead of silently overwriting partial
# progress (checkpoints, CSVs, train.log). The grid and effective flags are
# pinned in sweep.json at launch and checked on resume.

group_has_cells() {
  local d
  [ -d "$GROUP_DIR" ] || return 1
  for d in "$GROUP_DIR"/*/; do
    [ -d "$d" ] && return 0
  done
  return 1
}

latest_ckpt_step() { # $1 = run dir; echoes the max params_<step>.pkl step
  local f s best=""
  for f in "$1"/params_*.pkl; do
    [ -e "$f" ] || continue
    f="${f##*/params_}"
    s="${f%.pkl}"
    case "$s" in '' | *[!0-9]*) continue ;; esac
    if [ -z "$best" ] || [ "$s" -gt "$best" ]; then best="$s"; fi
  done
  printf '%s' "$best"
}

cell_state() { # $1 = cell dir base (alpha-<a>_seed-<s>); done|resume:<step>|fresh
  local d step best=""
  for d in "$GROUP_DIR/$1" "$GROUP_DIR/$1_slurm"*; do
    [ -d "$d" ] || continue
    if [ -f "$d/token.tk" ]; then
      printf 'done'
      return 0
    fi
    step="$(latest_ckpt_step "$d")"
    if [ -n "$step" ] && { [ -z "$best" ] || [ "$step" -gt "$best" ]; }; then
      best="$step"
    fi
  done
  if [ -n "$best" ]; then printf 'resume:%s' "$best"; else printf 'fresh'; fi
}

jget() { # $1 = sweep.json key; scalar raw, list comma-joined
  python3 -c '
import json, sys
v = json.load(open(sys.argv[1])).get(sys.argv[2])
if isinstance(v, list):
    print(",".join(map(str, v)))
elif v is None:
    print("")
else:
    print(v)
' "$SWEEP_JSON" "$1"
}

jget_lines() { # $1 = sweep.json key; one list item per line
  python3 -c '
import json, sys
for v in json.load(open(sys.argv[1])).get(sys.argv[2], []):
    print(v)
' "$SWEEP_JSON" "$1"
}

is_subset_csv() { # $1 needles, $2 haystack (comma-separated)
  local IFS=, needle
  for needle in $1; do
    case ",$2," in *,"$needle",*) ;; *) return 1 ;; esac
  done
}

check_resume_compat() {
  local want
  want="$(jget task)"
  [ "$want" = "$TASK_ID" ] || die "--resume: task $TASK_ID is not the stored $want"
  want="$(jget run_group)"
  [ "$want" = "$RUN_GROUP" ] || die "--resume: group $RUN_GROUP is not the stored $want"
  want="$(jget dataset)"
  [ "$want" = "$DATASET" ] || die "--resume: dataset $DATASET is not the stored $want"
  want="$(jget dataset_mtime)"
  [ "$want" = "$DS_MTIME" ] || die "--resume: dataset changed since this sweep started; start a new group instead"
  want="$(jget dataset_size)"
  [ "$want" = "$DS_SIZE" ] || die "--resume: dataset changed since this sweep started; start a new group instead"
  if ! diff <(jget_lines argv_tokens | sort | uniq -c) \
    <(printf '%s\n' "${EFF_ARGV[@]}" | sort | uniq -c) >/dev/null; then
    die "--resume: effective main.py flags differ from $SWEEP_JSON (wrapper edited?). Compare and start a new group, or restore the flags."
  fi
  local sha
  sha="$(jget qc_sha)"
  if [ -n "$sha" ] && [ -n "$QC_SHA" ] && [ "$sha" != "$QC_SHA" ]; then
    echo "[sweep] WARNING: qc checkout $QC_SHA differs from stored $sha; resuming across code changes." >&2
  fi
}

write_sweep_json() {
  local tmp="$GROUP_DIR/.sweep.json.tmp.$$" tok="$GROUP_DIR/.sweep.tokens.tmp.$$"
  printf '%s\n' "${EFF_ARGV[@]}" >"$tok"
  TASK="$TASK_ID" DATASET="$DATASET" MTIME="$DS_MTIME" SIZE="$DS_SIZE" \
    ALPHAS="$ALPHAS" SEEDS="$(IFS=,; echo "${seeds[*]}")" GROUP="$RUN_GROUP" \
    TOKFILE="$tok" QCSHA="$QC_SHA" OUTFILE="$tmp" python3 - <<'PY'
import json, os
toks = open(os.environ['TOKFILE']).read().splitlines()
doc = {
    'run_group': os.environ['GROUP'],
    'task': os.environ['TASK'],
    'dataset': os.environ['DATASET'],
    'dataset_mtime': os.environ['MTIME'],
    'dataset_size': os.environ['SIZE'],
    'alphas': os.environ['ALPHAS'].split(','),
    'seeds': [int(s) for s in os.environ['SEEDS'].split(',')],
    'argv_tokens': toks,
    'qc_sha': os.environ.get('QCSHA') or None,
    'save_interval': 50000,
}
open(os.environ['OUTFILE'], 'w').write(json.dumps(doc, indent=2) + '\n')
PY
  rm -f "$tok"
  mv "$tmp" "$SWEEP_JSON"
}

if [ "$RESUME" -eq 1 ] && [ "$OVERWRITE" -eq 1 ]; then
  die "--resume and --overwrite contradict each other"
fi

if [ "$RESUME" -eq 1 ]; then
  [ "$NUM_SEEDS_GIVEN" -eq 0 ] || die "--resume draws no fresh seeds; drop --num-seeds (the grid comes from $SWEEP_JSON)"
  if [ -f "$SWEEP_JSON" ]; then
    STORED_ALPHAS="$(jget alphas)"
    STORED_SEEDS="$(jget seeds)"
    [ -n "$STORED_ALPHAS" ] && [ -n "$STORED_SEEDS" ] ||
      die "$SWEEP_JSON has no grid; cannot --resume"
    if [ "$ALPHAS_GIVEN" -eq 1 ]; then
      is_subset_csv "$ALPHAS" "$STORED_ALPHAS" ||
        die "--alphas $ALPHAS is not a subset of the stored grid $STORED_ALPHAS"
    else
      ALPHAS="$STORED_ALPHAS"
      IFS=',' read -r -a alphas <<<"$ALPHAS"
    fi
    if [ "$SEEDS_GIVEN" -eq 1 ]; then
      is_subset_csv "$SEEDS" "$STORED_SEEDS" ||
        die "--seeds $SEEDS is not a subset of the stored grid $STORED_SEEDS"
      IFS=',' read -r -a seeds <<<"$SEEDS"
      NUM_SEEDS="${#seeds[@]}"
    else
      IFS=',' read -r -a seeds <<<"$STORED_SEEDS"
      NUM_SEEDS="${#seeds[@]}"
    fi
    [ "$DRY_RUN" -eq 1 ] || check_resume_compat
  else
    group_has_cells || die "nothing to --resume in $GROUP_DIR (no sweep.json, no cells)"
    echo "[sweep] WARNING: no sweep.json; resuming grandfather group without grid/flag checks." >&2
    [ "$SEEDS_GIVEN" -eq 1 ] || die "grandfather --resume needs explicit --seeds (stored seeds are unknown)"
    [ "$ALPHAS_GIVEN" -eq 1 ] || die "grandfather --resume needs explicit --alphas"
  fi
  RESUME_TAIL=(--resume)
else
  RESUME_TAIL=()
  if [ "$OVERWRITE" -eq 0 ] && [ "$DRY_RUN" -eq 0 ] && group_has_cells; then
    die "$GROUP_DIR already holds runs; relaunching would overwrite them. Pass --resume to continue, --overwrite to start over, or --run-group NAME for a separate group."
  fi
  if [ "$OVERWRITE" -eq 1 ] && [ "$DRY_RUN" -eq 0 ] && [ -d "$GROUP_DIR" ]; then
    rm -rf "$GROUP_DIR"
  fi
fi

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

# --- the kerberos ticket -----------------------------------------------------

# $REPO, the venvs and the dataset are read over NFS mounted with sec=krb5p, and
# a sweep outlives a login ticket. When the ticket expires the runs do not fail
# gracefully: they die with SIGBUS (status 135, the .venv pages they mmap'd
# stop being readable) or status 126 (train_qc.sh can no longer be executed).
# So the sweep refuses to start without a renewable ticket, renews it from a
# detached watcher that outlives the sweep itself, and treats 126/135 as infra
# failure: one costs a probe and a relaunch, a second stops the queue instead
# of launching the rest into the same wall.
RENEW_INTERVAL="${SWEEP_RENEW_INTERVAL:-1800}"
RENEW_PID=""
RENEW_LOG=""
WATCH_PIDFILE=""
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

# The watcher's body. Runs detached (setsid, stdio on local disk), so it
# survives the supervisor: the sweep's own bash is read lazily off the same
# NFS mount and dies with the ticket ("error reading input file") while
# orphaned runs live on. Everything arrives by environment: SWEEP_PID,
# WATCH_GROUP (the run_group to look for), RENEW_INTERVAL, RENEW_LOG,
# WATCH_PIDFILE.
_sweep_renew_loop() {
  set +e +u
  echo "$$" >"$WATCH_PIDFILE"
  rlog() { printf '[renew %s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" >>"$RENEW_LOG"; }
  expiry() { klist 2>/dev/null | awk 'NR==5 {print $3, $4}'; }

  rlog "watching sweep $SWEEP_PID, group $WATCH_GROUP, every ${RENEW_INTERVAL}s"
  command -v pgrep >/dev/null 2>&1 ||
    rlog "WARNING: no pgrep; orphaned runs cannot be detected"

  dead_rounds=0
  while :; do
    sleep "$RENEW_INTERVAL"
    if ! kill -0 "$SWEEP_PID" 2>/dev/null; then
      # The sweep is gone but orphaned runs may live on: stay only while a
      # run of this group is still alive.
      if command -v pgrep >/dev/null 2>&1 && pgrep -f "run_group=$WATCH_GROUP" >/dev/null 2>&1; then
        rlog "sweep gone but runs live on; keeping the ticket"
      else
        rlog "sweep gone and no runs left; exiting"
        return 0
      fi
    fi
    if kinit -R >>"$RENEW_LOG" 2>&1; then
      dead_rounds=0
      rlog "renewed, ticket expires $(expiry)"
      continue
    fi
    # Renewal is proactive (every 30 min against a 10 h ticket), so one miss
    # is a blip: two more tries a minute apart before calling it a round lost.
    tries=1
    while [ "$tries" -lt 3 ] && ! kinit -R >>"$RENEW_LOG" 2>&1; do
      tries=$((tries + 1))
      sleep 60
    done
    if [ "$tries" -lt 3 ]; then
      dead_rounds=0
      rlog "renewed after retry, ticket expires $(expiry)"
      continue
    fi
    dead_rounds=$((dead_rounds + 1))
    rlog "WARNING: kinit -R failed 3x (dead round $dead_rounds/3)"
    if [ "$dead_rounds" -ge 3 ]; then
      rlog "ticket unrecoverable; stopping sweep $SWEEP_PID"
      kill -TERM "$SWEEP_PID" 2>/dev/null || true
      return 1
    fi
  done
}

start_renewer() {
  [ "$RENEW_INTERVAL" != 0 ] || {
    repo_needs_ticket &&
      echo "[sweep] WARNING: renewal off (SWEEP_RENEW_INTERVAL=0) on a Kerberized mount; the runs die with the ticket." >&2
    return 0
  }
  repo_needs_ticket || return 0
  command -v kinit >/dev/null 2>&1 || die "$REPO needs a Kerberos ticket, but kinit is missing"
  command -v setsid >/dev/null 2>&1 || die "setsid is missing; cannot detach the ticket watcher"
  # Refused here rather than hours in, when the first run touches the share.
  klist -s 2>/dev/null || die "no valid Kerberos ticket for $REPO; run kinit first"
  klist 2>/dev/null | grep -q 'renew until' ||
    die "the ticket is not renewable; run 'kinit -r 14d' before a long sweep"
  # PAM deletes its own cache when any session of this user logs out, including
  # one that has nothing to do with the sweep; renewing does not bring it back.
  case "${KRB5CCNAME:-FILE:$PAM_CCACHE}" in
  *"$PAM_CCACHE")
    die "KRB5CCNAME is PAM's $PAM_CCACHE, deleted on any logout; use a private ccache (see ~/.bashrc) and kinit again"
    ;;
  esac
  mkdir -p "$GROUP_DIR"
  RENEW_LOG="$GROUP_DIR/krb-renew.log"
  WATCH_PIDFILE="$GROUP_DIR/.krb-renew.pid"
  # Detached, stdio on local disk: it must not die with the supervisor. The
  # pidfile carries the real pid across the setsid fork.
  export -f _sweep_renew_loop
  SWEEP_PID="$$" WATCH_GROUP="$RUN_GROUP" RENEW_INTERVAL="$RENEW_INTERVAL" \
    RENEW_LOG="$RENEW_LOG" WATCH_PIDFILE="$WATCH_PIDFILE" \
    setsid bash -c '_sweep_renew_loop' _sweep_renewer \
    </dev/null >>"$RENEW_LOG" 2>&1 &
  disown 2>/dev/null || true
  i=0
  while [ ! -s "$WATCH_PIDFILE" ] && [ "$i" -lt 50 ]; do
    sleep 0.1
    i=$((i + 1))
  done
  [ -s "$WATCH_PIDFILE" ] || die "ticket watcher failed to start; see $RENEW_LOG"
  RENEW_PID="$(cat "$WATCH_PIDFILE")"
  echo "[sweep] ticket   renewed every ${RENEW_INTERVAL}s by detached pid $RENEW_PID, log $RENEW_LOG"
}

# Both traps call this before stop_rest, whose bare `wait` would otherwise wait
# for a renewer that never returns.
stop_renewer() {
  [ -n "$RENEW_PID" ] || return 0
  kill "$RENEW_PID" 2>/dev/null || true
  wait "$RENEW_PID" 2>/dev/null || true
  RENEW_PID=""
  if [ -n "${WATCH_PIDFILE:-}" ]; then rm -f "$WATCH_PIDFILE"; fi
}

# --- launch ------------------------------------------------------------------

start_run() {
  local alpha="$1" seed="$2" log_dir tags
  log_dir="$GROUP_DIR/alpha-${alpha}_seed-$seed"
  tags="$(
    IFS=,
    echo "${SWEEP_TAGS[*]},alpha-$alpha,seed-$seed"
  )"

  # The defaults come first, then the wrapper's flags, then the flags typed on
  # top of it, so each stage can override the one before it (absl takes the
  # last occurrence). The sweep's own flags come last: --seed, the alpha,
  # the group and --resume are what the sweep *is*, and are not overridable
  # from either.
  set -- "$TASK_ID" "$DATASET" \
    "${DEFAULT_MAIN_FLAGS[@]}" \
    ${BASE_FLAGS[@]+"${BASE_FLAGS[@]}"} \
    ${USER_FLAGS[@]+"${USER_FLAGS[@]}"} \
    --seed="$seed" \
    --agent.alpha="$alpha" \
    --run_group="$RUN_GROUP" \
    --wandb_tags="$tags" \
    ${RESUME_TAIL[@]+"${RESUME_TAIL[@]}"}

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$RESUME" -eq 1 ]; then
      echo "[dry-run] alpha $alpha seed $seed -> $(cell_state "alpha-${alpha}_seed-${seed}"), log $log_dir/train.log"
    else
      echo "[dry-run] alpha $alpha seed $seed -> $log_dir/train.log"
    fi
    printf '[dry-run]   %s/train_qc.sh' "$REPO"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  mkdir -p "$log_dir"
  if [ "$RESUME" -eq 1 ] && [ -f "$log_dir/train.log" ]; then
    # A resumed cell keeps its history: truncating would destroy the log of
    # the run these checkpoints continue. Fresh cells (or fresh sweeps) start
    # clean. This also covers the 126/135 relaunch below, which re-enters here.
    "$REPO/train_qc.sh" "$@" >>"$log_dir/train.log" 2>&1 &
  else
    "$REPO/train_qc.sh" "$@" >"$log_dir/train.log" 2>&1 &
  fi
  LAST_PID="$!" # the caller's handle on the run just started
  pids+=("$LAST_PID")
  cell_of["$LAST_PID"]="alpha-$alpha seed-$seed"
  echo "[sweep] alpha $alpha seed $seed started, pid $LAST_PID, log $log_dir/train.log"
}

# Seed-major, so one seed runs across all alphas before the next seed.
queue=()
for seed in "${seeds[@]}"; do
  for alpha in "${alphas[@]}"; do
    queue+=("$alpha $seed")
  done
done

if [ "$RESUME" -eq 1 ]; then
  kept=()
  for cell in "${queue[@]}"; do
    alpha="${cell%% *}"
    seed="${cell##* }"
    base="alpha-${alpha}_seed-${seed}"
    state="$(cell_state "$base")"
    case "$state" in
    done)
      echo "[sweep] alpha-$alpha seed-$seed done, skipping"
      ;;
    resume:*)
      echo "[sweep] alpha-$alpha seed-$seed resumes at step ${state#resume:}"
      kept+=("$cell")
      ;;
    fresh)
      echo "[sweep] alpha-$alpha seed-$seed no checkpoint, fresh start"
      kept+=("$cell")
      ;;
    esac
  done
  queue=(${kept[@]+"${kept[@]}"})
  if [ "${#queue[@]}" -eq 0 ]; then
    echo "[sweep] group $RUN_GROUP: all runs done"
    exit 0
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  if [ "$RESUME" -eq 0 ] && [ "$OVERWRITE" -eq 0 ] && group_has_cells; then
    echo "[dry-run] NOTE: $GROUP_DIR exists; a real launch would abort without --resume/--overwrite." >&2
  fi
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
INFRA_DOWN=0 # a 126/135 means the ticket died: drain, then stop the queue
declare -A retried=() # cells already relaunched once after a 126/135
LAST_PID="" # set by start_run, read by its caller right after

# Ctrl-C stops every run of the sweep.
stop_rest() {
  for pid in ${running[@]+"${running[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'echo "[sweep] stopping, killing the runs" >&2; stop_renewer; stop_rest; exit 130' INT TERM
trap stop_renewer EXIT

start_renewer

# One sweeper per group: two concurrent sweeps would launch the same cells
# twice, interleaving CSVs and clobbering checkpoints.
mkdir -p "$GROUP_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$GROUP_DIR/.sweep.lock" || die "cannot lock $GROUP_DIR"
  flock -n 9 || die "$GROUP_DIR is locked by another sweep; refusing to run concurrently"
else
  echo "[sweep] WARNING: flock is missing; concurrent sweeps on one group are unsafe." >&2
fi

# Pin the grid and flags for a later --resume. Resume mode keeps the stored
# file: it describes the group, not the invocation (a subset resume must not
# narrow it).
if [ "$RESUME" -eq 0 ]; then
  write_sweep_json
fi

reap_one() {
  local status=0 done_pid="" rest=()
  wait -n -p done_pid ${running[@]+"${running[@]}"} || status=$?
  local cell="${cell_of[${done_pid:-}]:-?}"
  for pid in ${running[@]+"${running[@]}"}; do
    [ "$pid" = "${done_pid:-}" ] || rest+=("$pid")
  done
  running=(${rest[@]+"${rest[@]}"})
  if [ "$status" -ne 0 ]; then
    case "$status" in
    126 | 135)
      if [ -z "${retried[$cell]:-}" ]; then
        # A seconds-long NFS blip looks exactly like a dead ticket, so one
        # death buys a 60 s probe and a relaunch; the cell did no work (the
        # exec itself failed), overwriting its train.log is safe.
        retried[$cell]=1
        echo "[sweep] $cell died with status $status; probing the mount for 60s before calling it infra failure" >&2
        sleep 60 || true
        if [ -x "$REPO/train_qc.sh" ]; then
          echo "[sweep] $cell: mount readable again (a blip, not the ticket); relaunching" >&2
          alpha="${cell%% *}"
          alpha="${alpha#alpha-}"
          seed="${cell##*seed-}"
          start_run "$alpha" "$seed"
          running+=("$LAST_PID")
          echo "[sweep] ${#running[@]} run(s) going, $((${#queue[@]} - next)) waiting"
          return 0
        fi
      fi
      echo "[sweep] $cell died with status $status: NFS unreadable, the Kerberos ticket is dead -- not the experiment" >&2
      INFRA_DOWN=1
      ;;
    *)
      echo "[sweep] $cell failed with status $status, see $GROUP_DIR/${cell// /_}/train.log" >&2
      ;;
    esac
    failed+=("$cell")
  else
    echo "[sweep] $cell done"
  fi
  echo "[sweep] ${#running[@]} run(s) going, $((${#queue[@]} - next)) waiting"
}

next=0
while [ "$next" -lt "${#queue[@]}" ] || [ "${#running[@]}" -gt 0 ]; do
  while [ "$next" -lt "${#queue[@]}" ] && [ "${#running[@]}" -lt "$MAX_PARALLEL" ] && [ "$INFRA_DOWN" -eq 0 ]; do
    # shellcheck disable=SC2086  # a queue entry is "<alpha> <seed>", split on purpose
    start_run ${queue[$next]}
    running+=("$LAST_PID")
    next=$((next + 1))
  done
  reap_one
done

if [ "$INFRA_DOWN" -ne 0 ]; then
  echo "[sweep] group $RUN_GROUP: stopped early: ticket dead (status 126/135). Run 'kinit -r 14d' and relaunch with --run-group NAME." >&2
  exit 1
fi
if [ "${#failed[@]}" -gt 0 ]; then
  echo "[sweep] group $RUN_GROUP: failed: ${failed[*]}" >&2
  exit 1
fi
echo "[sweep] group $RUN_GROUP: all runs done"
