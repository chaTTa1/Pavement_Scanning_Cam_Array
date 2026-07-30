#!/usr/bin/env bash
#
# run_agx.sh — start the two AGX-side programs in the correct order.
#
#   1) remote_display_raw_bytes_v2.py  (orchestrator: SSH-launches the nanos,
#      receives images, and publishes the shared pulse epoch to PULSE_EPOCH_FILE)
#   2) CV7_read_nmea (1)_v2.py         (reads the shared pulse epoch, stamps
#      every row's T_from_pulse against the same pulse the images use)
#
# remote_display MUST start first so the pulse-epoch file exists before CV7
# begins logging. This script waits for that file before launching CV7.
#
# Ctrl-C stops BOTH programs cleanly.
#
# Usage:
#   ./run_agx.sh                 # normal run
#   ./run_agx.sh --help          # show options
#
# Any extra args after "--cv7" are forwarded to the CV7 script, e.g.:
#   ./run_agx.sh --cv7 --port /dev/ttyACM0 --output /data/run1
#
set -u

# ----------------------------------------------------------------------------
# Config — edit these to match your machine
# ----------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"

# Directory this script lives in (scripts are assumed to sit next to it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DISPLAY_SCRIPT="${DISPLAY_SCRIPT:-$SCRIPT_DIR/remote_display_raw_bytes_v2.py}"
CV7_SCRIPT="${CV7_SCRIPT:-$SCRIPT_DIR/CV7_read_nmea (1)_v2.py}"

# Shared pulse-epoch file written by remote_display, read by CV7.
PULSE_EPOCH_FILE="${PULSE_EPOCH_FILE:-/tmp/pulse_epoch.json}"

# How long (seconds) to wait for the pulse-epoch file before starting CV7
# anyway. Set to 0 to start CV7 immediately (early rows may have blank
# T_from_pulse until the first image arrives).
EPOCH_WAIT_TIMEOUT="${EPOCH_WAIT_TIMEOUT:-120}"

# Where to put per-run logs.
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

# ----------------------------------------------------------------------------
# Arg parsing: everything after "--cv7" goes to the CV7 script.
# ----------------------------------------------------------------------------
CV7_ARGS=()
DISPLAY_ARGS=()
_target="display"
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    --cv7) _target="cv7"; continue ;;
    --display) _target="display"; continue ;;
  esac
  if [[ "$_target" == "cv7" ]]; then CV7_ARGS+=("$arg"); else DISPLAY_ARGS+=("$arg"); fi
done

# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: '$PYTHON' not found."; exit 1; }
[[ -f "$DISPLAY_SCRIPT" ]] || { echo "ERROR: display script not found: $DISPLAY_SCRIPT"; exit 1; }
[[ -f "$CV7_SCRIPT"     ]] || { echo "ERROR: CV7 script not found: $CV7_SCRIPT"; exit 1; }

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
DISPLAY_LOG="$LOG_DIR/remote_display_$STAMP.log"
CV7_LOG="$LOG_DIR/cv7_$STAMP.log"

DISPLAY_PID=""
CV7_PID=""

cleanup() {
  echo ""
  echo "[run_agx] shutting down..."
  [[ -n "$CV7_PID"     ]] && kill "$CV7_PID"     2>/dev/null
  [[ -n "$DISPLAY_PID" ]] && kill "$DISPLAY_PID" 2>/dev/null
  # give them a moment, then force
  sleep 2
  [[ -n "$CV7_PID"     ]] && kill -9 "$CV7_PID"     2>/dev/null
  [[ -n "$DISPLAY_PID" ]] && kill -9 "$DISPLAY_PID" 2>/dev/null
  echo "[run_agx] done. Logs in: $LOG_DIR"
}
trap cleanup INT TERM EXIT

# ----------------------------------------------------------------------------
# 1) Start remote_display (also launches the nanos over SSH)
# ----------------------------------------------------------------------------
# Remove any stale epoch file from a previous run so we wait for a fresh one.
rm -f "$PULSE_EPOCH_FILE" 2>/dev/null

echo "[run_agx] starting remote_display -> $DISPLAY_LOG"
"$PYTHON" "$DISPLAY_SCRIPT" "${DISPLAY_ARGS[@]}" 2>&1 | tee "$DISPLAY_LOG" &
DISPLAY_PID=$!

# ----------------------------------------------------------------------------
# 2) Wait for the shared pulse-epoch file, then start CV7
# ----------------------------------------------------------------------------
if [[ "$EPOCH_WAIT_TIMEOUT" -gt 0 ]]; then
  echo "[run_agx] waiting up to ${EPOCH_WAIT_TIMEOUT}s for $PULSE_EPOCH_FILE ..."
  waited=0
  while [[ ! -f "$PULSE_EPOCH_FILE" && "$waited" -lt "$EPOCH_WAIT_TIMEOUT" ]]; do
    # bail out early if remote_display died
    if ! kill -0 "$DISPLAY_PID" 2>/dev/null; then
      echo "[run_agx] ERROR: remote_display exited before publishing the pulse epoch."
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  if [[ -f "$PULSE_EPOCH_FILE" ]]; then
    echo "[run_agx] pulse epoch published: $(cat "$PULSE_EPOCH_FILE")"
  else
    echo "[run_agx] WARNING: epoch file not seen after ${EPOCH_WAIT_TIMEOUT}s; starting CV7 anyway."
    echo "[run_agx]          early CV7 rows may have a blank T_from_pulse."
  fi
fi

echo "[run_agx] starting CV7 -> $CV7_LOG"
"$PYTHON" "$CV7_SCRIPT" "${CV7_ARGS[@]}" 2>&1 | tee "$CV7_LOG" &
CV7_PID=$!

echo "[run_agx] both running. PIDs: display=$DISPLAY_PID cv7=$CV7_PID"
echo "[run_agx] press Ctrl-C to stop both."

# ----------------------------------------------------------------------------
# Wait: if either child exits, tear everything down.
# ----------------------------------------------------------------------------
while true; do
  if ! kill -0 "$DISPLAY_PID" 2>/dev/null; then
    echo "[run_agx] remote_display exited."; break
  fi
  if ! kill -0 "$CV7_PID" 2>/dev/null; then
    echo "[run_agx] CV7 exited."; break
  fi
  sleep 1
done
