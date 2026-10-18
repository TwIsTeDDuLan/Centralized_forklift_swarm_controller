#!/usr/bin/env bash
# ==============================================================================
# Centralized Forklift Swarm - Motion & Capability Demonstration Runner
# ==============================================================================
# Usage:
#   ./run_demo.sh            # Launch interactive demo menu
#   ./run_demo.sh --auto     # Run full synchronized choreography (all 3 forklifts)
#   ./run_demo.sh --forward  # Drive forward
#   ./run_demo.sh --backward # Drive backward
#   ./run_demo.sh --lift-up  # Raise forks
#   ./run_demo.sh --lift-down# Lower forks
#   ./run_demo.sh --stop     # Emergency stop
# ==============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${ROS_DISTRO:-}" ]] || ! command -v ros2 >/dev/null 2>&1; then
    if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
        set +eu
        # shellcheck disable=SC1090
        source "/opt/ros/jazzy/setup.bash"
        set -eo pipefail
    fi
fi

export PYTHONUNBUFFERED=1
python3 demo_swarm_motion.py "$@"
