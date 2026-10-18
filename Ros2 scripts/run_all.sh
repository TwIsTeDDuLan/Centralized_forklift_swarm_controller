#!/usr/bin/env bash
# ==============================================================================
# Centralized Forklift Swarm - Unified Runner (Multi-Terminal Launcher)
# ==============================================================================
# Description:
#   Orchestrates the entire swarm stack:
#     1. Vision Tracker (vision_tracker.py via run_tracker.sh)
#     2. Fleet Controller (fleet_controller.py via run_fleet_controller.sh)
#     3. Forklift 1 Node (Fork_Lift_1.py via run_forklift_1.sh)
#     4. Forklift 2 Node (Fork_Lift_2.py via run_forklift_2.sh)
#     5. Forklift 3 Node (Fork_Lift_3.py via run_forklift_3.sh)
#   Launches each component into separate dedicated terminal windows or tabs.
#
# Usage:
#   ./run_all.sh                  # Launch all 5 components in separate windows
#   ./run_all.sh --tabs           # Launch all 5 components as tabs in one window
#   ./run_all.sh --no-forklifts   # Launch only Tracker & Controller
#   ./run_all.sh --only-forklifts # Launch only the 3 Forklift nodes
#   ./run_all.sh --check          # Preflight health check for all components
#   ./run_all.sh --help           # Show this help message
# ==============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    echo "[WARN] Not running in bash. Re-executing with bash..." >&2
    exec bash "$0" "$@"
fi

set -eo pipefail

# --- Color formatting ---
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
Usage: ./run_all.sh [OPTIONS]

Component Selection:
  -a, --all                Launch all 5 components: Tracker, Controller, Forklifts 1-3 (default)
  --no-forklifts           Launch only Vision Tracker and Fleet Controller
  --only-forklifts         Launch only the 3 Forklift nodes (Forklift 1, 2, 3)
  -t, --tracker            Launch only the Vision Tracker
  -f, --controller         Launch only the Fleet Controller
  --fl1                    Launch only Forklift 1
  --fl2                    Launch only Forklift 2
  --fl3                    Launch only Forklift 3

Terminal Display Modes:
  -w, --windows            Open each component in a separate terminal window (default)
  --tabs                   Open components in tabs within a single terminal window
  --same-term, --parallel  Run concurrently inside this terminal (for SSH/headless)

Options:
  --no-aruco               Launch tracker with ArUco detection disabled
  -c, --check, --dry-run   Run full preflight diagnostic checks without launching
  -v, --verbose            Enable verbose diagnostic output
  -d, --debug              Enable debug tracing
  -h, --help               Display this help guide

Examples:
  ./run_all.sh                     # Launch all 5 components in separate windows
  ./run_all.sh --tabs              # Launch all 5 components in tabs
  ./run_all.sh --no-forklifts      # Launch Tracker + Fleet Controller only
  ./run_all.sh --only-forklifts    # Launch Forklifts 1, 2, and 3 only
  ./run_all.sh --check             # Check environment for all scripts and exit
EOF
    exit 0
}

# --- Parse arguments ---
LAUNCH_TRACKER=1
LAUNCH_CONTROLLER=1
LAUNCH_FL1=1
LAUNCH_FL2=1
LAUNCH_FL3=1
TERMINAL_MODE="windows"
CHECK_ONLY=0
NO_ARUCO=0
VERBOSE_MODE=${VERBOSE:-0}
DEBUG_MODE=${DEBUG:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -a|--all)
            LAUNCH_TRACKER=1
            LAUNCH_CONTROLLER=1
            LAUNCH_FL1=1
            LAUNCH_FL2=1
            LAUNCH_FL3=1
            shift
            ;;
        --no-forklifts)
            LAUNCH_TRACKER=1
            LAUNCH_CONTROLLER=1
            LAUNCH_FL1=0
            LAUNCH_FL2=0
            LAUNCH_FL3=0
            shift
            ;;
        --only-forklifts)
            LAUNCH_TRACKER=0
            LAUNCH_CONTROLLER=0
            LAUNCH_FL1=1
            LAUNCH_FL2=1
            LAUNCH_FL3=1
            shift
            ;;
        -t|--tracker)
            LAUNCH_TRACKER=1
            LAUNCH_CONTROLLER=0
            LAUNCH_FL1=0
            LAUNCH_FL2=0
            LAUNCH_FL3=0
            shift
            ;;
        -f|--controller)
            LAUNCH_TRACKER=0
            LAUNCH_CONTROLLER=1
            LAUNCH_FL1=0
            LAUNCH_FL2=0
            LAUNCH_FL3=0
            shift
            ;;
        --fl1|--forklift-1)
            LAUNCH_TRACKER=0
            LAUNCH_CONTROLLER=0
            LAUNCH_FL1=1
            LAUNCH_FL2=0
            LAUNCH_FL3=0
            shift
            ;;
        --fl2|--forklift-2)
            LAUNCH_TRACKER=0
            LAUNCH_CONTROLLER=0
            LAUNCH_FL1=0
            LAUNCH_FL2=1
            LAUNCH_FL3=0
            shift
            ;;
        --fl3|--forklift-3)
            LAUNCH_TRACKER=0
            LAUNCH_CONTROLLER=0
            LAUNCH_FL1=0
            LAUNCH_FL2=0
            LAUNCH_FL3=1
            shift
            ;;
        -w|--windows)
            TERMINAL_MODE="windows"
            shift
            ;;
        --tabs)
            TERMINAL_MODE="tabs"
            shift
            ;;
        --same-term|--same-terminal|--parallel)
            TERMINAL_MODE="same"
            shift
            ;;
        --no-aruco)
            NO_ARUCO=1
            shift
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
        *)
            log_warn "Unrecognized option: $1"
            shift
            ;;
    esac
done

if [[ "${DEBUG_MODE}" -eq 1 ]]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TRACKER_SCRIPT="${SCRIPT_DIR}/run_tracker.sh"
CONTROLLER_SCRIPT="${SCRIPT_DIR}/run_fleet_controller.sh"
FL1_SCRIPT="${SCRIPT_DIR}/run_forklift_1.sh"
FL2_SCRIPT="${SCRIPT_DIR}/run_forklift_2.sh"
FL3_SCRIPT="${SCRIPT_DIR}/run_forklift_3.sh"

# Ensure runner scripts exist and are executable
ALL_SCRIPTS=("${TRACKER_SCRIPT}" "${CONTROLLER_SCRIPT}" "${FL1_SCRIPT}" "${FL2_SCRIPT}" "${FL3_SCRIPT}")
for s in "${ALL_SCRIPTS[@]}"; do
    if [[ ! -f "$s" ]]; then
        log_err "Required script not found: $s"
        exit 1
    fi
    chmod +x "$s"
done

# --- Diagnostic Check Mode ---
if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
    echo -e "${C_BOLD}${C_CYAN}         Centralized Forklift Swarm - Unified Diagnostics             ${C_RESET}"
    echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
    echo ""
    log_info "1/5 Running Vision Tracker diagnostics..."
    "${TRACKER_SCRIPT}" --check
    echo ""
    log_info "2/5 Running Fleet Controller diagnostics..."
    "${CONTROLLER_SCRIPT}" --check
    echo ""
    log_info "3/5 Checking Forklift 1..."
    "${FL1_SCRIPT}" --check
    echo ""
    log_info "4/5 Checking Forklift 2..."
    "${FL2_SCRIPT}" --check
    echo ""
    log_info "5/5 Checking Forklift 3..."
    "${FL3_SCRIPT}" --check
    echo ""
    log_ok "All component diagnostics passed successfully!"
    exit 0
fi

# --- Detect Available Terminal Emulator ---
find_terminal() {
    if [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
        return 1
    fi
    for term in gnome-terminal xterm xfce4-terminal konsole tilix alacritty kitty; do
        if command -v "$term" >/dev/null 2>&1; then
            echo "$term"
            return 0
        fi
    done
    return 1
}

TERM_BIN="$(find_terminal || true)"

if [[ -z "${TERM_BIN}" ]] && [[ "${TERMINAL_MODE}" != "same" ]]; then
    log_warn "No graphical terminal emulator or active display found. Falling back to same-terminal mode."
    TERMINAL_MODE="same"
fi

# Build safe command strings with proper single-quoting for directories with spaces
build_term_cmd() {
    local script_name="$1"
    shift
    local cmd="cd '${SCRIPT_DIR}' && ./'${script_name}'"
    if [[ $# -gt 0 ]]; then
        cmd+=" $*"
    fi
    echo "${cmd}"
}

TRACKER_ARGS=()
if [[ "${NO_ARUCO}" -eq 1 ]]; then
    TRACKER_ARGS+=("--no-aruco")
fi
if [[ "${VERBOSE_MODE}" -eq 1 ]]; then
    TRACKER_ARGS+=("--verbose")
fi
if [[ "${DEBUG_MODE}" -eq 1 ]]; then
    TRACKER_ARGS+=("--debug")
fi

TRACKER_CMD_STR="$(build_term_cmd "run_tracker.sh" "${TRACKER_ARGS[@]}")"
CONTROLLER_CMD_STR="$(build_term_cmd "run_fleet_controller.sh")"
FL1_CMD_STR="$(build_term_cmd "run_forklift_1.sh")"
FL2_CMD_STR="$(build_term_cmd "run_forklift_2.sh")"
FL3_CMD_STR="$(build_term_cmd "run_forklift_3.sh")"

# --- Banner ---
echo ""
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}         Centralized Forklift Swarm - Multi-Terminal Launcher         ${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo -e "  • Terminal Emulator: $([[ -n "${TERM_BIN}" ]] && echo "${TERM_BIN}" || echo "None (Same Terminal)")"
echo -e "  • Launch Mode:       ${TERMINAL_MODE}"
echo -e "  • Vision Tracker:    $([[ ${LAUNCH_TRACKER} -eq 1 ]] && echo "${C_GREEN}ENABLED${C_RESET}" || echo "${C_GRAY}SKIPPED${C_RESET}")"
echo -e "  • Fleet Controller:  $([[ ${LAUNCH_CONTROLLER} -eq 1 ]] && echo "${C_GREEN}ENABLED${C_RESET}" || echo "${C_GRAY}SKIPPED${C_RESET}")"
echo -e "  • Forklift 1 Node:   $([[ ${LAUNCH_FL1} -eq 1 ]] && echo "${C_GREEN}ENABLED${C_RESET}" || echo "${C_GRAY}SKIPPED${C_RESET}")"
echo -e "  • Forklift 2 Node:   $([[ ${LAUNCH_FL2} -eq 1 ]] && echo "${C_GREEN}ENABLED${C_RESET}" || echo "${C_GRAY}SKIPPED${C_RESET}")"
echo -e "  • Forklift 3 Node:   $([[ ${LAUNCH_FL3} -eq 1 ]] && echo "${C_GREEN}ENABLED${C_RESET}" || echo "${C_GRAY}SKIPPED${C_RESET}")"
echo -e "  • ArUco Tracking:    $([[ ${NO_ARUCO} -eq 1 ]] && echo "${C_YELLOW}DISABLED${C_RESET}" || echo "${C_GREEN}ENABLED${C_RESET}")"
echo -e "  • Working Directory: ${SCRIPT_DIR}"
echo -e "${C_BOLD}${C_CYAN}======================================================================${C_RESET}"
echo ""

# --- Launch Logic ---

launch_in_gnome_terminal_tabs() {
    log_info "Launching swarm stack in tabs using gnome-terminal..."
    local first_tab=1

    # gnome-terminal accepts only one "-- command" per invocation, so each tab needs its own call.
    add_tab() {
        local title="$1"
        local cmd="$2"
        local mode="--tab"
        if [[ ${first_tab} -eq 1 ]]; then
            mode="--window"
            first_tab=0
        fi
        gnome-terminal "${mode}" --title="${title}" --working-directory="${SCRIPT_DIR}" \
            -- bash -c "${cmd}; echo ''; echo -e '${C_GRAY}Process ended. Press Enter to close...${C_RESET}'; read -r"
        sleep 0.5
    }

    [[ ${LAUNCH_TRACKER} -eq 1 ]] && add_tab "[Swarm] Vision Tracker" "${TRACKER_CMD_STR}"
    [[ ${LAUNCH_CONTROLLER} -eq 1 ]] && add_tab "[Swarm] Fleet Controller" "${CONTROLLER_CMD_STR}"
    [[ ${LAUNCH_FL1} -eq 1 ]] && add_tab "[Swarm] Forklift 1" "${FL1_CMD_STR}"
    [[ ${LAUNCH_FL2} -eq 1 ]] && add_tab "[Swarm] Forklift 2" "${FL2_CMD_STR}"
    [[ ${LAUNCH_FL3} -eq 1 ]] && add_tab "[Swarm] Forklift 3" "${FL3_CMD_STR}"
    return 0
}

launch_in_separate_windows() {
    log_info "Opening separate terminal windows for active components..."

    open_window() {
        local title="$1"
        local cmd="$2"
        log_info "Opening ${title}..."
        if [[ "${TERM_BIN}" == "gnome-terminal" ]]; then
            gnome-terminal \
                --title="${title}" \
                --working-directory="${SCRIPT_DIR}" \
                -- bash -c "${cmd}; echo ''; echo -e '${C_GRAY}Process ended. Press Enter to close...${C_RESET}'; read -r" &
        elif [[ "${TERM_BIN}" == "xterm" ]]; then
            xterm -title "${title}" -e bash -c "${cmd}; echo ''; echo 'Press Enter to close...'; read -r" &
        else
            "${TERM_BIN}" -e bash -c "${cmd}; echo ''; echo 'Press Enter to close...'; read -r" &
        fi
    }

    [[ ${LAUNCH_TRACKER} -eq 1 ]] && open_window "[Swarm] Vision Tracker" "${TRACKER_CMD_STR}"
    [[ ${LAUNCH_CONTROLLER} -eq 1 ]] && open_window "[Swarm] Fleet Controller" "${CONTROLLER_CMD_STR}"
    [[ ${LAUNCH_FL1} -eq 1 ]] && open_window "[Swarm] Forklift 1" "${FL1_CMD_STR}"
    [[ ${LAUNCH_FL2} -eq 1 ]] && open_window "[Swarm] Forklift 2" "${FL2_CMD_STR}"
    [[ ${LAUNCH_FL3} -eq 1 ]] && open_window "[Swarm] Forklift 3" "${FL3_CMD_STR}"
}

launch_same_terminal() {
    log_info "Running components concurrently in current terminal..."
    log_info "Press [Ctrl+C] to stop all processes cleanly."
    echo -e "${C_GRAY}----------------------------------------------------------------------${C_RESET}"

    PIDS=()
    cleanup_all() {
        echo ""
        log_info "Stopping all swarm processes..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait 2>/dev/null || true
        log_ok "All swarm processes stopped."
        exit 0
    }
    trap cleanup_all SIGINT SIGTERM

    [[ ${LAUNCH_TRACKER} -eq 1 ]] && { "${TRACKER_SCRIPT}" "${TRACKER_ARGS[@]}" 2>&1 | sed -u "s/^/${C_BLUE}[TRACKER]${C_RESET}    /" & PIDS+=($!); }
    [[ ${LAUNCH_CONTROLLER} -eq 1 ]] && { "${CONTROLLER_SCRIPT}" 2>&1 | sed -u "s/^/${C_MAGENTA}[CONTROLLER]${C_RESET} /" & PIDS+=($!); }
    [[ ${LAUNCH_FL1} -eq 1 ]] && { "${FL1_SCRIPT}" 2>&1 | sed -u "s/^/${C_CYAN}[FORKLIFT-1]${C_RESET} /" & PIDS+=($!); }
    [[ ${LAUNCH_FL2} -eq 1 ]] && { "${FL2_SCRIPT}" 2>&1 | sed -u "s/^/${C_GREEN}[FORKLIFT-2]${C_RESET} /" & PIDS+=($!); }
    [[ ${LAUNCH_FL3} -eq 1 ]] && { "${FL3_SCRIPT}" 2>&1 | sed -u "s/^/${C_YELLOW}[FORKLIFT-3]${C_RESET} /" & PIDS+=($!); }

    wait
}

case "${TERMINAL_MODE}" in
    tabs)
        if [[ "${TERM_BIN}" == "gnome-terminal" ]]; then
            launch_in_gnome_terminal_tabs
        else
            log_warn "Tabs mode only supported on gnome-terminal. Falling back to separate windows."
            launch_in_separate_windows
        fi
        ;;
    same)
        launch_same_terminal
        ;;
    windows|*)
        launch_in_separate_windows
        ;;
esac

if [[ "${TERMINAL_MODE}" != "same" ]]; then
    echo ""
    log_ok "All requested component terminals have been launched!"
    echo -e "  • Use the dedicated terminal windows to view live output."
    echo -e "  • Terminals stay open when stopped so you can inspect logs anytime."
fi
