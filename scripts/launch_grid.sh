#!/usr/bin/env bash
# Launch one grid in a detached tmux session pinned to one GPU.
#
#   scripts/launch_grid.sh <session> <gpu> <grid.yaml> [extra train.py args]
#
# Each grid row is a separate train.py subprocess run SEQUENTIALLY, so one
# session owns one GPU end-to-end. Rows whose eval_report.json already exists
# are skipped, which makes a session safe to re-launch after a crash.
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,9p' "$0"; exit 2
fi

SESSION=$1; GPU=$2; GRID=$3; shift 3

REPO=/home/data/zhengwenbo/hyper_mve
PY=/home/zhengwenbo/.conda/envs/lightzero/bin/python
LOGDIR="$REPO/results/_logs"

case "$GPU" in
    3|4|5|6) ;;
    *) echo "refusing GPU '$GPU': training is restricted to GPUs 3,4,5,6" >&2
       exit 2 ;;
esac
[ -f "$REPO/$GRID" ] || [ -f "$GRID" ] || { echo "no such grid: $GRID" >&2; exit 2; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists — inspect it with:" >&2
    echo "    tmux attach -t $SESSION" >&2
    exit 1
fi

mkdir -p "$LOGDIR"
LOG="$LOGDIR/${SESSION}_$(date +%Y%m%d_%H%M%S).log"

# `tee` keeps the scrollback in tmux AND on disk; `; exec bash` holds the pane
# open after the grid finishes so a non-zero exit is still readable.
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "$PY scripts/train.py --grid '$GRID' --gpus $GPU $* 2>&1 | tee '$LOG'; exec bash"

echo "launched '$SESSION'  gpu=$GPU  grid=$GRID"
echo "  follow : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
echo "  log    : tail -f $LOG"
