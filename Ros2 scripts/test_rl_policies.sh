#!/usr/bin/env bash
# ==============================================================================
# Centralized Forklift Swarm - RL Policy Test Runner
# ==============================================================================
# Run this in a separate terminal while run_all.sh is active.
#
# Usage:
#   ./test_rl_policies.sh                    # Interactive test menu
#   ./test_rl_policies.sh --check            # Preflight check of camera & forklifts
#   ./test_rl_policies.sh --nav 1 640 360 90 # Test Go-To-Pose on forklift 1
#   ./test_rl_policies.sh --pick 1 1 0.3     # Test pallet pick (stage -> insert -> lift)
#   ./test_rl_policies.sh --cycle 1 1        # Full pick -> deliver -> place cycle
#   ./test_rl_policies.sh --place 1          # Put down pallet and back out
#   ./test_rl_policies.sh --monitor          # Live stream of swarm poses & statuses
#   ./test_rl_policies.sh --stop [all|1|2|3] # Stop forklift(s)
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Always source ROS 2 Jazzy environment
if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
else
    echo "[ERROR] ROS 2 Jazzy (/opt/ros/jazzy/setup.bash) not found!" >&2
    exit 1
fi

# Ensure python test harness is executable
chmod +x "${SCRIPT_DIR}/test_rl_policies.py" 2>/dev/null || true

# Execute test suite
exec python3 "${SCRIPT_DIR}/test_rl_policies.py" "$@"
