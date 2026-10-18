#!/usr/bin/env bash
# ==============================================================================
# Centralized Forklift Swarm - Fleet Controller Runner & Troubleshooting Tool
# ==============================================================================
# Description:
#   Sets up the ROS 2 Jazzy environment, validates all system & Python
#   dependencies, and launches the central fleet controller node with real-time
#   verbose diagnostic output.
#
# Usage:
#   ./run_fleet_controller.sh              # Run fleet controller with verbose logs
#   ./run_fleet_controller.sh --check      # Preflight diagnostic check (dry-run)
#   ./run_fleet_controller.sh --verbose    # Enable extra debugging output
#   ./run_fleet_controller.sh --debug      # Enable bash xtrace (set -x) & env dump
#   ./run_fleet_controller.sh --help       # Show this help message
# ==============================================================================

# Ensure execution under bash if invoked via sh
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[WARN] Not running in bash. Re-executing with bash..." >&2
    exec bash "$0" "$@"
fi

set -eo pipefail

# --- Color formatting (auto-disabled if non-interactive or NO_COLOR set) ---
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    C_RESET='\033[0m'
    C_BOLD='\033[1m'
    C_RED='\033[0;31m'
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[0;33m'
    C_BLUE='\033[0;34m'
    C_CYAN='\033[0;36m'
    C_MAGENTA='\033[0;35m'
    C_GRAY='\033[0;90m'
else
    C_RESET=''
    C_BOLD=''
    C_RED=''
    C_GREEN=''
    C_YELLOW=''
    C_BLUE=''
    C_CYAN=''
    C_MAGENTA=''
    C_GRAY=''
fi

# --- Logging helpers ---
timestamp() {
    date +"%H:%M:%S"
}

log_info() {
    echo -e "${C_CYAN}[INFO]${C_RESET}  $(timestamp) $*"
}

log_ok() {
    echo -e "${C_GREEN}[OK]${C_RESET}    $(timestamp) $*"
}

log_warn() {
    echo -e "${C_YELLOW}[WARN]${C_RESET}  $(timestamp) $*" >&2
}

log_err() {
    echo -e "${C_RED}[ERROR]${C_RESET} $(timestamp) $*" >&2
}

log_debug() {
    if [[ "${VERBOSE_MODE:-0}" -eq 1 ]] || [[ "${DEBUG_MODE:-0}" -eq 1 ]]; then
        echo -e "${C_GRAY}[DEBUG]${C_RESET} $(timestamp) $*"
    fi
}

show_help() {
    cat << 'EOF'
Usage: ./run_fleet_controller.sh [OPTIONS] [-- ROS_ARGS]

Options:
  -c, --check, --dry-run   Run environment & dependency diagnostics without starting node
  -v, --verbose            Enable verbose output and topic diagnostics
  -d, --debug              Enable full debug mode (bash xtrace and environment dump)
  -h, --help               Display this help guide

Environment Variables:
  ROS_DISTRO               ROS 2 distribution (default: jazzy)
  ROS_SETUP_FILE           Custom path to setup.bash (default: /opt/ros/jazzy/setup.bash)
  PYTHON_BIN               Custom Python 3 binary path (default: autodetected)
  ROS_DOMAIN_ID            ROS 2 Domain ID (default: 0)
  VERBOSE                  Set to 1 to enable verbose output
  DEBUG                    Set to 1 to enable debug output

Examples:
  ./run_fleet_controller.sh
  ./run_fleet_controller.sh --check
  ./run_fleet_controller.sh --verbose
  VERBOSE=1 ./run_fleet_controller.sh
EOF
    exit 0
}

# --- Parse arguments ---
CHECK_ONLY=0
VERBOSE_MODE=${VERBOSE:-0}
DEBUG_MODE=${DEBUG:-0}
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -c|--check|--dry-run)
            CHECK_ONLY=1
            shift
            ;;
        -v|--verbose)
            VERBOSE_MODE=1
            shift
            ;;
        -d|--debug)
            DEBUG_MODE=1
            VERBOSE_MODE=1
            shift
            ;;
        --)
            shift
            FORWARD_ARGS+=("$@")
            break
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${DEBUG_MODE}" -eq 1 ]]; then
    set -x
fi

# --- 1. Working Directory Check ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
log_info "Working directory set to: ${SCRIPT_DIR}"

# Verify fleet_controller.py exists
TARGET_SCRIPT="${SCRIPT_DIR}/fleet_controller.py"
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    log_err "Target controller script not found: ${TARGET_SCRIPT}"
    exit 1
fi
log_ok "Located target script: fleet_controller.py"

# --- 2. Locate and Source ROS 2 ---
find_ros_setup() {
    # 1. Custom user override
    if [[ -n "${ROS_SETUP_FILE:-}" ]] && [[ -f "${ROS_SETUP_FILE}" ]]; then
        echo "${ROS_SETUP_FILE}"
        return 0
    fi
    # 2. Preferred default: ROS 2 Jazzy
    if [[ -f "/opt/ros/jazzy/setup.bash" ]]; then
        echo "/opt/ros/jazzy/setup.bash"
        return 0
    fi
    # 3. Check current ROS_DISTRO if defined
    if [[ -n "${ROS_DISTRO:-}" ]] && [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
        echo "/opt/ros/${ROS_DISTRO}/setup.bash"
        return 0
    fi
    # 4. Search any installed /opt/ros/*/setup.bash
    for candidate in /opt/ros/*/setup.bash; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

ROS_SETUP=""
# Check if ROS 2 is already active
if [[ -n "${ROS_DISTRO:-}" ]] && command -v ros2 >/dev/null 2>&1 && [[ -n "${AMENT_PREFIX_PATH:-}" ]]; then
    log_ok "ROS 2 is already active in current environment (ROS_DISTRO=${ROS_DISTRO})"
else
    ROS_SETUP="$(find_ros_setup || true)"
    if [[ -z "${ROS_SETUP}" ]]; then
        log_err "No ROS 2 setup.bash found!"
        echo -e "  Checked: /opt/ros/jazzy/setup.bash and other /opt/ros/*/setup.bash"
        echo -e "  Troubleshooting:"
        echo -e "    1. Ensure ROS 2 (e.g., Jazzy) is installed on your system."
        echo -e "    2. Or specify ROS_SETUP_FILE manually, e.g.:"
        echo -e "       ROS_SETUP_FILE=/path/to/setup.bash ./run_fleet_controller.sh"
        exit 1
    fi

    log_info "Sourcing ROS 2 setup: ${ROS_SETUP}"
    # Temporarily allow unset vars / errors during setup sourcing
    set +eu
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
    set -eo pipefail
    log_ok "ROS 2 environment sourced successfully (ROS_DISTRO=${ROS_DISTRO:-jazzy})"
fi

# Ensure LD_LIBRARY_PATH includes ROS 2 lib directory so C-extensions (_rclpy_pybind11) load
ROS_LIB_PATH="/opt/ros/${ROS_DISTRO:-jazzy}/lib"
if [[ -d "${ROS_LIB_PATH}" ]]; then
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":${ROS_LIB_PATH}:"* ]]; then
        export LD_LIBRARY_PATH="${ROS_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        log_debug "Appended ${ROS_LIB_PATH} to LD_LIBRARY_PATH"
    fi
fi

# --- 3. Locate Python Interpreter ---
find_python() {
    if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        command -v "${PYTHON_BIN}"
        return 0
    fi
    if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python3" ]]; then
        echo "${VIRTUAL_ENV}/bin/python3"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if [[ -x "/usr/bin/python3" ]]; then
        echo "/usr/bin/python3"
        return 0
    fi
    return 1
}

PYTHON_CMD="$(find_python || true)"
if [[ -z "${PYTHON_CMD}" ]]; then
    log_err "Python 3 interpreter could not be found!"
    exit 1
fi

PY_VERSION="$("${PYTHON_CMD}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null || echo "unknown")"
log_ok "Python interpreter located: ${PYTHON_CMD} (v${PY_VERSION})"

# --- 4. Validate Required Python Dependencies ---
log_info "Verifying ROS 2 Python dependencies..."
MISSING_MODULES=()

for mod in rclpy geometry_msgs ackermann_msgs sensor_msgs numpy; do
    if ! "${PYTHON_CMD}" -c "import ${mod}" >/dev/null 2>&1; then
        MISSING_MODULES+=("${mod}")
        log_err "  Module check: ${mod} -> [MISSING / FAILED]"
    else
        log_debug "  Module check: ${mod} -> [OK]"
    fi
done

if [[ ${#MISSING_MODULES[@]} -gt 0 ]]; then
    log_err "Preflight dependency check failed! Missing: ${MISSING_MODULES[*]}"
    echo -e "  Troubleshooting guidance:"
    for m in "${MISSING_MODULES[@]}"; do
        case "$m" in
            rclpy)
                echo -e "    -> sudo apt install ros-${ROS_DISTRO:-jazzy}-rclpy"
                ;;
            geometry_msgs)
                echo -e "    -> sudo apt install ros-${ROS_DISTRO:-jazzy}-geometry-msgs"
                ;;
            ackermann_msgs)
                echo -e "    -> sudo apt install ros-${ROS_DISTRO:-jazzy}-ackermann-msgs"
                ;;
            sensor_msgs)
                echo -e "    -> sudo apt install ros-${ROS_DISTRO:-jazzy}-sensor-msgs"
                ;;
            numpy)
                echo -e "    -> sudo apt install python3-numpy (or pip3 install numpy)"
                ;;
        esac
    done
    exit 1
fi
log_ok "All required modules verified (rclpy, geometry_msgs, ackermann_msgs, sensor_msgs, numpy)"

# --- 5. Diagnostic Dry-Run Mode ---
if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo ""
    echo -e "${C_BOLD}${C_GREEN}======================================================================${C_RESET}"
    echo -e "${C_BOLD}${C_GREEN}                DIAGNOSTIC PREFLIGHT CHECK: PASSED                    ${C_RESET}"
    echo -e "${C_BOLD}${C_GREEN}======================================================================${C_RESET}"
    echo -e "  • ROS 2 Distro:      ${ROS_DISTRO:-unknown}"
    echo -e "  • ROS 2 Prefix:      ${AMENT_PREFIX_PATH:-unknown}"
    echo -e "  • ROS Domain ID:     ${ROS_DOMAIN_ID:-0 (default)}"
    echo -e "  • Python Executable: ${PYTHON_CMD} (v${PY_VERSION})"
    echo -e "  • Script Path:       ${TARGET_SCRIPT}"
    echo -e "  • Working Directory: ${SCRIPT_DIR}"
    echo ""
    log_ok "Your environment is healthy and ready to run the Fleet Controller."
    exit 0
fi

# --- 6. Execution Banner ---
echo ""
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}           Centralized Forklift Swarm - Fleet Controller              ${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "  • ROS 2 Distro:      ${ROS_DISTRO:-jazzy} (Domain ID: ${ROS_DOMAIN_ID:-0})"
echo -e "  • Python:            ${PYTHON_CMD} (v${PY_VERSION})"
echo -e "  • Controller:        fleet_controller.py"
echo -e "  • Mode:              Active (Press ${C_BOLD}Ctrl+C${C_RESET} to stop)"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo ""

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1

# Trap interrupts for a clean terminal shutdown
cleanup() {
    echo ""
    log_info "Caught termination signal. Exiting fleet controller..."
}
trap cleanup SIGINT SIGTERM

# --- 7. Launch Fleet Controller ---
set +e
"${PYTHON_CMD}" fleet_controller.py "${FORWARD_ARGS[@]}"
EXIT_CODE=$?
set -e

echo ""
echo -e "${C_GRAY}----------------------------------------------------------------------${C_RESET}"
if [[ ${EXIT_CODE} -eq 0 ]] || [[ ${EXIT_CODE} -eq 130 ]]; then
    log_ok "Fleet Controller terminated cleanly."
else
    log_warn "Fleet Controller exited with status code: ${EXIT_CODE}"
    echo -e "  Troubleshooting tip:"
    echo -e "    Run './run_fleet_controller.sh --check' to verify the environment."
    echo -e "    Run './run_fleet_controller.sh --debug' for full shell trace."
fi
echo -e "${C_GRAY}----------------------------------------------------------------------${C_RESET}"

exit ${EXIT_CODE}
