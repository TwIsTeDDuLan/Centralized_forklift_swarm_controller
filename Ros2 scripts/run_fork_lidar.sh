#!/usr/bin/env bash
# Attach the 3x64 fork lidar after Environment.usd is already playing.
# Does not edit any USD. Publishes /forklift_N/scan for the insert policy.
#
# Preferred: Isaac Sim Script Editor → File → Open attach_fork_lidar.py → Run
# This script only prints that path (and optionally tries isaaclab.sh -p if you pass --kit).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIDAR="${SCRIPT_DIR}/attach_fork_lidar.py"

echo "Fork lidar must run inside the already-open Isaac Sim stage."
echo "  1. Environment.usd is playing"
echo "  2. Window → Script Editor"
echo "  3. File → Open:"
echo "       ${LIDAR}"
echo "  4. Run"
echo "You should see /forklift_1/scan (192 ranges) on ros2 topic list."
echo ""

if [[ "${1:-}" == "--kit" ]]; then
    ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
    # Only works if this process shares the Kit stage; usually it does not.
    echo "Trying isaaclab.sh -p (needs the warehouse already loaded in this Kit)..."
    source "${ISAACLAB}/.venv/bin/activate"
    exec "${ISAACLAB}/isaaclab.sh" -p "${LIDAR}"
fi
