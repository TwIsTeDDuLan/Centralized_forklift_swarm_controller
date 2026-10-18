#!/usr/bin/env bash
# Train the forklift go-to-pose policy in the background (keeps running if the terminal closes).
#   ./scripts/train.sh                                # 1024 envs, 1000 iterations, no viewer (Isaac Lab 3 default)
#   ./scripts/train.sh --num_envs 2048 --max_iterations 1500
#   ./scripts/train.sh --resume --load_run <run_dir_name>   # continue a stopped run
#   FOREGROUND=1 ./scripts/train.sh --num_envs 16 --max_iterations 3   # quick smoke test
# Extra arguments go to Isaac Lab's unified train script.
#
# Isaac Lab writes logs/ relative to the working directory, so running from isaac_lab_forklift/ keeps all
# checkpoints, TensorBoard events and the console log on the Local Disk, not the NVMe.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}
TASK=${TASK:-Isaac-Forklift-Nav-Direct}
mkdir -p "$HERE/logs"
LOG="$HERE/logs/train_${TASK}.log"
cd "$HERE"
# isaaclab.sh -p puts Isaac Sim on the path; it needs the venv active
source "$ISAACLAB/.venv/bin/activate"

CMD=("$ISAACLAB/isaaclab.sh" -p "$ISAACLAB/scripts/reinforcement_learning/train.py"
     --rl_library rsl_rl --task "$TASK" --num_envs 1024 "$@")

if [[ -n "${FOREGROUND:-}" ]]; then
    exec "${CMD[@]}"
fi
setsid nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
echo "training $TASK (pid $!), log: $LOG"
echo "progress:     tail -f \"$LOG\" | grep -E --line-buffered 'Learning iteration|Mean reward|success_rate|curriculum'"
echo "tensorboard:  $ISAACLAB/.venv/bin/tensorboard --logdir \"$HERE/logs/rsl_rl/forklift_nav\""
