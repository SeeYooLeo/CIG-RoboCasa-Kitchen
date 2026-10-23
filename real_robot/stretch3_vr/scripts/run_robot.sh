#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Use the robot's installed ROS workspace and fleet configuration.
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROBOT_WS="${ROBOT_WS:-$HOME/ament_ws}"
set +u
source "$ROS_SETUP"
source "$ROBOT_WS/install/setup.bash"
set -u

STAGE="${STAGE:-base_locked}"
OVERLAY_ARGS=()
if [[ -n "${SITE_OVERLAY:-}" ]]; then
  OVERLAY_ARGS+=(--config-overlay "$SITE_OVERLAY")
fi
case "$STAGE" in
  base_locked) OVERLAY_ARGS+=(--config-overlay config/overlay_base_locked.json) ;;
  slow) OVERLAY_ARGS+=(--config-overlay config/overlay_slow.json) ;;
  full) ;;
  *) echo "STAGE must be base_locked, slow or full" >&2; exit 2 ;;
esac
CERT="${CERT:-certs/server.crt}"
KEY="${KEY:-certs/server.key}"
RECORD_ARGS=()
if [[ -n "${RECORD_DIR:-}" ]]; then
  RECORD_ARGS+=(--record-dir "$RECORD_DIR")
fi
.venv/bin/stretch3-vr "${OVERLAY_ARGS[@]}" doctor --ros2 \
  --camera-probe-s 3 --cert "$CERT" --key "$KEY"
exec .venv/bin/stretch3-vr "${OVERLAY_ARGS[@]}" serve --backend ros2 \
  --host "${BIND_HOST:-0.0.0.0}" --port "${PORT:-8443}" --cert "$CERT" --key "$KEY" \
  "${RECORD_ARGS[@]}"
