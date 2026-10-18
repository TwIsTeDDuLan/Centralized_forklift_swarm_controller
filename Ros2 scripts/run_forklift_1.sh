#!/usr/bin/env bash
# ==============================================================================
# Centralized Forklift Swarm - Forklift 1 Runner & Diagnostics
# ==============================================================================
# Description:
#   Sources the ROS 2 Jazzy environment and launches Forklift 1
#   (Fork_Lift_1.py) with real-time verbose output.
#
# Usage:
#   ./run_forklift_1.sh              # Run Forklift 1 node
#   ./run_forklift_1.sh --check      # Preflight diagnostic check
#   ./run_forklift_1.sh --help       # Display help
# ==============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    echo "[WARN] Not running in bash. Re-executing with bash..." >&2
    exec bash "$0" "$@"
fi

set -eo pipefail

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    C_RESET='\033[0m'
    C_BOLD='\033[1m'
    C_RED='\033[0;31m'
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[0;33m'
    C_CYAN='\033[0;36m'
    C_GRAY='\033[0;90m'
else
    C_RESET=''
    C_BOLD=''
    C_RED=''
    C_GREEN=''
    C_YELLOW=''
    C_CYAN=''
    C_GRAY=''
fi

timestamp() {
    date +"%H:%M:%S"
}

log_info() {
    echo -e "${C_CYAN}[INFO]${C_RESET}  $(timestamp) $*"
}

log_ok() {
    echo -e "${C_GREEN}[OK]${C_RESET}    $(timestamp) $*"
}

log_err() {
    echo -e "${C_RED}[ERROR]${C_RESET} $(timestamp) $*" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TARGET_SCRIPT="${SCRIPT_DIR}/Fork Lifts/Fork_Lift_1.py"
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    log_err "Target script not found: ${TARGET_SCRIPT}"
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
    log_ok "Forklift 1 script verified: ${TARGET_SCRIPT}"
    log_ok "ROS 2 environment ready: ROS_DISTRO=${ROS_DISTRO:-jazzy}"
    exit 0
fi

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: ./run_forklift_1.sh [--check | --help]"
    exit 0
fi

echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}                 Forklift 1 Robot Controller                          ${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "  • Script:       Fork Lifts/Fork_Lift_1.py"
echo -e "  • ROS Distro:   ${ROS_DISTRO:-jazzy}"
echo -e "  • Subscribes:   /forklift_1/lift_target_pos (Pose2D)"
echo -e "  • Subscribes:   /forklift_1/mast_target_pos (Twist)"
echo -e "  • Mode:         Active (Press Ctrl+C to stop)"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo ""

export PYTHONUNBUFFERED=1

cleanup() {
    echo ""
    log_info "Stopping Forklift 1..."
}
trap cleanup SIGINT SIGTERM

python3 "${TARGET_SCRIPT}" "$@"
