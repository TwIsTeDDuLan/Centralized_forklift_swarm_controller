#!/usr/bin/env bash
# ==============================================================================
# Centralized Forklift Swarm - Forklift 3 Runner & Diagnostics
# ==============================================================================
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -eo pipefail

C_CYAN='\033[0;36m'
C_RESET='\033[0m'
C_BOLD='\033[1m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TARGET_SCRIPT="${SCRIPT_DIR}/Fork Lifts/Fork_Lift_3.py"
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "[ERROR] Target script not found: ${TARGET_SCRIPT}" >&2
    exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]] || ! command -v ros2 >/dev/null 2>&1; then
    if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
        set +eu
        # shellcheck disable=SC1090
        source "/opt/ros/jazzy/setup.bash"
        set -eo pipefail
    fi
fi

if [[ "$1" == "--check" ]] || [[ "$1" == "-c" ]]; then
    echo "[OK] Forklift 3 script verified: ${TARGET_SCRIPT}"
    exit 0
fi

echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}                 Forklift 3 Robot Controller                          ${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "  • Script:       Fork Lifts/Fork_Lift_3.py"
echo -e "  • ROS Distro:   ${ROS_DISTRO:-jazzy}"
echo -e "  • Subscribes:   /forklift_3/lift_target_pos (Pose2D)"
echo -e "  • Subscribes:   /forklift_3/mast_target_pos (Twist)"
echo -e "  • Mode:         Active (Press Ctrl+C to stop)"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo ""

export PYTHONUNBUFFERED=1
trap "echo ''; echo '[INFO] Stopping Forklift 3...'" SIGINT SIGTERM
python3 "${TARGET_SCRIPT}" "$@"
