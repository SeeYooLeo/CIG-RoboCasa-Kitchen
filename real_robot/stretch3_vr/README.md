# Stretch 3 VR teleoperation

[Project overview](../../README.md)

This module provides the VR teleoperation interface used to collect demonstrations for the paper: a Quest 3S WebXR client, a Python WebSocket server, and a Stretch 3 ROS 2 backend. The operator controls the mobile base, head, arm, wrist and gripper while viewing head/navigation and wrist camera streams.

The module supports timestamped VR episode recording and replay. Its `labels.json` container shares the outer structure of a legacy recorder, while VR actions retain explicit ROS 2 units and are replayed through ROS 2. The optional legacy adapter supports older gamepad-format files and their `stretch_body` units; it is not the paper's data-collection interface. The four robocasa-kitchen checkpoints target the simulator and are not mapped to Stretch hardware. No recorded demonstrations or real-robot success-rate claim are included.

## Paper experiment settings

The paper uses VR-collected demonstrations for all three Stretch 3 kitchen tasks, with 50 demonstrations per task and an average episode duration of approximately two minutes. The experiment settings describe the processed VR dataset and the subsequent policy deployment:

| Stage | Paper setting |
|---|---|
| Demonstration interface | Quest 3S VR headset / WebXR / Stretch 3 ROS 2 |
| Aligned demonstration rate | 15 Hz after timestamp-based alignment |
| Head and wrist RGB | Center-cropped and resized to 224 × 224 × 3 |
| Policy state / action | 14D / 18D after dataset conversion |
| Processed video encoding | H.264, CRF 15 |
| Policy deployment | RTX 5090, 15 Hz |
| Action horizon / replan / skip | 12 / 5 / 3 steps |
| Held-out evaluation | 20 rollouts per task and method |

The raw VR recorder stores timestamped ROS commands, feedback and JPEG images. The default browser sends approximately 30 control frames/s; `robot.camera_encode_hz=15` limits camera JPEG encoding, `network.camera_send_hz=12` limits headset preview updates, and `robot.joint_command_hz=4` limits joint-goal submissions. These are distinct from the aligned policy-data rate. Preserve the recorded timestamps during replay and align the streams during downstream training-data preprocessing. The policy-data converter and real-robot policy deployment loop are outside this module.

## Requirements and installation

For real hardware, use a configured Stretch 3 running Ubuntu 22.04 and ROS 2 Humble, with the robot's calibrated `stretch_ai` Pinocchio model and `stretch_ros2` workspace available. The backend imports `rclpy`, ROS action/message packages, `stretch.motion.kinematics`, NumPy, OpenCV and `cv_bridge`. Its IK adapter uses the manipulation solver's 9-DOF seed and the installed `stretch_ai` conversion helpers. Keep the robot's fleet identity and calibration in its existing environment.

Python 3.10 or newer is required. The mock backend and core control logic do not require ROS or robot hardware. Node.js is needed only for the panel geometry checks.

From the release root:

```bash
export ROOT="$PWD"
cd "$ROOT/real_robot/stretch3_vr"
./scripts/setup_env.sh
.venv/bin/stretch3-vr doctor
.venv/bin/stretch3-vr smoke
```

The setup script creates an editable installation in `.venv` with access to the system's ROS Python packages. Run commands from this module directory; keep `config/` beside `src/`.

## Preview without robot hardware

```bash
.venv/bin/stretch3-vr serve --backend mock \
  --host 127.0.0.1 --port 8080 --insecure-http
```

Open `http://localhost:8080/` for the English UI and connection diagnostics. A desktop browser can preview the page; immersive control requires WebXR on the headset. The mock backend has no images by default. To preview camera panels, provide your own JPEG directory with `--camera-dir /path/to/images`; optional `head/`, `navigation/` and `gripper/` subdirectories supply separate views.

## HTTPS and headset connection

Quest Browser must treat the page as a secure context. Generate fresh local credentials for your robot's IPv4 address:

```bash
export ROBOT_IP="<robot-ip>"  # Replace with the robot's reachable IPv4 address.
./scripts/generate_dev_cert.sh "$ROBOT_IP"
```

Trust `certs/quest_ca.crt` as a local CA on the headset before opening `https://<robot-ip>:8443/`. A browser warning bypass alone may not make WebXR available; the page's diagnostics must report a secure context. Keep `quest_ca.key` and `server.key` on the server. Certificates and keys are generated locally and excluded from Git.

For an initial headset-only preview, run the mock backend with TLS:

```bash
.venv/bin/stretch3-vr serve --backend mock --host 0.0.0.0 --port 8443 \
  --cert certs/server.crt --key certs/server.key
```

Use a trusted, isolated operator network. The server accepts one controlling client and does not implement user authentication; TLS encrypts the connection but does not authorize an operator. Do not expose the control port to the public Internet.

## Robot setup and launch

Bring up the robot's configured ROS driver and camera nodes using its normal operating procedure. Only one driver/control stack should own the hardware. Homing and calibration remain part of that procedure; this module does not perform them automatically.

Example ROS commands, each in its own sourced terminal, if those nodes are not already running:

```bash
source /opt/ros/humble/setup.bash
export ROBOT_WS="$HOME/ament_ws"  # Adjust to your installed workspace.
source "$ROBOT_WS/install/setup.bash"
ros2 launch stretch_core stretch_driver.launch.py
```

```bash
source /opt/ros/humble/setup.bash
export ROBOT_WS="$HOME/ament_ws"  # Adjust in each terminal.
source "$ROBOT_WS/install/setup.bash"
ros2 launch stretch_core multi_camera.launch.py
```

The optional navigation camera launcher requests 1280x800 MJPG from `/dev/hello-nav-head-camera`. Use it only if this device and mode are available, and no other node owns the camera:

```bash
source /opt/ros/humble/setup.bash
export ROBOT_WS="$HOME/ament_ws"
source "$ROBOT_WS/install/setup.bash"
# Set ROOT to the release directory in this terminal.
ros2 launch "$ROOT/real_robot/stretch3_vr/launch/navigation_camera_wide.launch.py"
```

In the teleoperation terminal, run a read-only ROS/topic/camera check, then start the bridge:

```bash
cd "$ROOT/real_robot/stretch3_vr"
source /opt/ros/humble/setup.bash
export ROBOT_WS="$HOME/ament_ws"
source "$ROBOT_WS/install/setup.bash"
.venv/bin/stretch3-vr doctor --ros2 --camera-probe-s 3 \
  --cert certs/server.crt --key certs/server.key
STAGE=base_locked ./scripts/run_robot.sh
```

`base_locked` sets all commanded base speeds, including IK base compensation, to zero. **It still permits head, arm, wrist and gripper movement when their controls are activated; it is not a no-motion mode.** Keep the workspace clear and the physical emergency stop accessible. The launcher performs the diagnostic check before serving and never kills existing processes or homes the robot.

Once the robot-specific setup is verified, `STAGE=slow ./scripts/run_robot.sh` selects reduced base speeds and end-effector translation scaling. `STAGE=full` uses the default speed limits. `ROS_SETUP`, `ROBOT_WS`, `CERT`, `KEY`, `BIND_HOST` and `PORT` can override the launch paths and address. Stop the foreground process with Ctrl+C; release the controls before stopping it.

## Configuration and controls

`config/default.json` defines the ROS topics, controller limits, camera orientation, panel layout and input timeout. Apply robot-specific values through a local JSON overlay. For example:

```json
{
  "controls": {"operator_yaw_deg": -90.0},
  "robot": {
    "camera_rotate_deg": {"navigation": 270, "head": 90, "gripper": 0},
    "left_panel_by_mode": {"NAV": "navigation"}
  }
}
```

Save it as `config/local.json` and use `SITE_OVERLAY=config/local.json STAGE=slow ./scripts/run_robot.sh`. These angles are examples, not universal calibration. Verify camera orientation, operator coordinate mapping, head limits, gripper aperture and IK on your robot. The launcher applies the speed-stage overlay last. Local overlays are excluded from Git. Optional wrist depth cues are disabled by default and require aligned depth and an appropriate reference distance.

| Control | Default behavior |
|---|---|
| Left Grip | Hold to enable base and head control; release to stop the base |
| Left stick | Base translation and yaw |
| Left X, short press | NAV → ALIGN → MANIP; MANIP returns to ALIGN |
| Left X, long press | Return toward NAV through ALIGN |
| Right Grip | Acquire/release the end-effector anchor |
| Right Trigger | Close gripper; automatically enter PRECISION |
| Right A / B | Toggle MANIP/PRECISION / return to ALIGN |
| Left Y + Grip | Temporarily enable head control during manipulation |
| Right stick | Adjust panel distance/size; click to recenter |

PRECISION locks the base. The default input timeout is 0.3 seconds; an independent watchdog checks for missing valid control frames, including while non-control messages arrive. Timeout, disconnect, save and normal shutdown publish zero base velocity, cancel outstanding joint goals (including goals awaiting acceptance), and call the driver stop service. The driver must acknowledge cancellation and the stop request; a failure blocks further motion until the backend is restarted. Joint feedback older than 0.5 seconds also stops motion. These are software controls, not collision avoidance or a certified emergency-stop system. The depth overlay is a visual cue only.

Default camera topics are `/camera/color/image_raw`, `/navigation_camera/image_raw`, and `/gripper_camera/color/image_rect_raw`. The browser shows two panels: the configured head/navigation source on the left and the wrist camera on the right. `GET /healthz` reports backend, camera and transport diagnostics without taking the controlling client slot. Camera resolution and bandwidth are determined by the ROS drivers; the robocasa-kitchen 128→224 image pipeline does not apply to this live interface.

## Recording and replay

### Collect from the VR interface

Enable recording explicitly when launching the real backend:

```bash
RECORD_DIR="$ROOT/recordings/stretch3" STAGE=slow ./scripts/run_robot.sh
```

For an end-to-end trial without hardware:

```bash
.venv/bin/stretch3-vr serve --backend mock --host 127.0.0.1 \
  --port 8080 --insecure-http --record-dir "$ROOT/recordings/mock"
```

1. Enter the task instruction in the browser before entering VR.
2. Select **Start recording**, then perform the demonstration. In VR, release both Grips and click the left stick to start; a red square in each camera panel indicates recording.
3. Select **Save success** or **Save failure**. The same left-stick shortcut saves success while an episode is active. Saving requests a stop before finalizing the files. **Abort episode** retains an incomplete episode for diagnosis.
4. Inspect the saved episode using the replay command below. A missing `completed.txt` or `complete: false` prevents replay.

The `record_control` WebSocket messages are `{ "type": "record_control", "operation": "start", "task": "..." }` and `{ "type": "record_control", "operation": "save", "success": true }`; use `"abort"` to terminate without marking the episode complete. The server returns recording status to the controlling client and exposes it in `/healthz`. This reuses the single operator connection and does not launch another hardware controller.

A completed episode has:

```text
<episode-id>/
  configs.json        # Schema, task, backend, effective configuration, outcome
  labels.json         # Frames indexed by "0", "1", ...
  frames.jsonl        # Streaming recovery journal for this episode
  images/<camera>/    # JPEG samples; repeated images share a file reference
  completed.txt       # Present only for a successfully finalized recording
```

Each label contains `timestamp_s`, `observation_timestamp_s`, `submission_duration_s`, `observations`, `actions`, image paths/hashes/source stamps/ages, `client_sequence` and a `kind` such as `control`, `watchdog` or `stop`. Observations include joint feedback, base odometry, head pose and the available FK end-effector estimate. Joint and odometry ages are recorded separately. Actions contain:

```json
{
  "base_linear_mps": 0.02,
  "base_angular_rps": 0.0,
  "joint_targets": {"joint_lift": 0.5, "wrist_extension": 0.1},
  "trajectory_time_s": 0.2,
  "stop_motion": false
}
```

Base velocities are in m/s and rad/s. Lift, total arm extension (`wrist_extension`) and `gripper_aperture` are in meters; wrist and head targets are in radians. `joint_targets` contains only targets actually submitted at that control step, after IK and command-rate limiting. Empty targets mean no new joint trajectory was submitted. `stop_motion: true` explicitly cancels joint motion and stops the base; it must have zero base velocities and no joint targets. Replay requires this flag on stop/watchdog events, so recordings from an earlier implementation that omitted it must not be treated as validated stop sequences. Goal acceptance is checked on subsequent control steps and when stopping; a rejected goal prevents completion. Acceptance does not establish physical target arrival or task success. Compare the recorded feedback when assessing execution. The installed driver uses navigation-mode joint targets; `trajectory_time_s` is the submitted ROS field, not a guarantee of actuator travel time.

The VR client sends control frames at approximately 30 Hz, and the recorder retains every submitted control event plus stop/watchdog events. Raw event recording is timestamp-driven; the paper's 15 Hz timeline is formed during downstream dataset alignment. Actions are recorded as submitted commands rather than inferred state differences. Cameras are sampled asynchronously at their configured encoding rate, without the UI's depth overlay; images retain camera resolution and configured rotation. State and images are sampled before applying each action. `observation_timestamp_s` records that sample time; `timestamp_s` records the start of ROS command publication after IK, and `submission_duration_s` measures the complete apply call. Base publication and joint-goal submission are sequential, not atomic. These are latest-available observations with explicit timestamps/ages, not hardware-synchronized exposures. Missing images are never replaced with fabricated black frames. Real-backend recording requires fresh head/wrist frames at startup and becomes incomplete if either is missing or older than 0.5 s. The navigation stream is included when available.

A bounded background writer keeps disk I/O out of the control loop. Saving seals the event queue synchronously before background finalization, so watchdog events cannot follow the terminal stop. Episode duration excludes file-flush time. Overflow, write errors, connection loss and shutdown leave the episode incomplete. Private metadata such as Git history, hostname and fleet identifiers is not collected. The default format stores JPEGs, not MP4 videos or raw depth; downstream training-format conversion is separate.

### Replay a VR episode

```bash
# Validate indices, timestamps, actions, completion and image checksums; no motion.
.venv/bin/stretch3-vr replay /path/to/episode

# Replay the submitted command sequence in a mock backend using recorded timing.
.venv/bin/stretch3-vr replay /path/to/episode --execute --backend mock

# Robot operation: stop the VR server first, retain the ROS driver/cameras,
# and manually prepare the same robot configuration and start pose.
.venv/bin/stretch3-vr replay /path/to/episode --execute --backend ros2
```

Replay uses the saved post-IK joint targets and base velocities, without solving IK again or guessing a fixed frame rate. It checks the initial feedback for commanded non-gripper joints, does not automatically move to the start pose, and stops if command scheduling falls more than 0.25 s behind, including oversleep while waiting for the next timestamp. A final stop is applied on completion or exception. Mock recordings cannot be executed on the real backend. The package's VR and replay entry points share an exclusive hardware-control lock; independently launched controllers must also be stopped.

Replay is open-loop command reproduction, not a guarantee of the same physical path or task outcome. Prepare the same base location/heading, gripper state, scene and objects; initial joint checks do not verify these conditions. Keep the physical stop available. Use Ctrl+C to interrupt.

### Legacy compatibility: existing gamepad recordings

This optional compatibility path reads older `FileDataRecorder` episodes containing `labels.json`, `configs.json`, and video files. It is separate from the paper's VR demonstration collection. The legacy gamepad-format adapter preserves two action modes:

```bash
# Inspect a legacy episode. FPS comes from configs.json, or an explicit --fps.
.venv/bin/stretch3-vr legacy-replay /path/to/legacy-episode

# Direct stretch_body replay: stop the VR server AND the ROS robot driver first.
# Keep the ROS environment sourced so the ownership check can query the graph.
.venv/bin/stretch3-vr legacy-replay /path/to/legacy-episode \
  --mode velocity --execute
```

`velocity` reissues saved `v_*` commands; `target` reissues absolute joint targets and converts base increments using the saved FPS. Both preserve the original gripper `move_by` actuator convention. Legacy gripper values are not ROS aperture meters. This adapter must run alone, checks for active ROS driver/VR nodes, requires a homed robot and matching initial joint pose, and does not hardcode a fleet identity or move automatically to a start pose. Legacy recordings lack per-frame timestamps, so their timing can only be estimated from FPS. Do not change FPS to claim faithful playback.

The separate Dex leader uses `base_x_joint` and a different action schema; its files are explicitly rejected by this gamepad adapter rather than silently interpreted as velocities. The VR recording path replaces the input source while retaining explicit, replayable ROS command semantics.

## Validation

Run the hardware-free checks:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -p no:cacheprovider
node tests/panel_geometry.mjs
.venv/bin/stretch3-vr smoke
```

The tests cover recording/save/load/mock-replay round trips, image integrity, incomplete episodes, writer failures, legacy action units, exclusive ownership, control transitions, input validation, timeout behavior, WebSocket transport, camera routing, depth cues and panel geometry. Mock tests do not establish hardware compatibility or validate motion on a new robot.

The release contains source, default configurations, launch helpers and tests. Runtime logs, fleet identifiers, calibration files, certificates, local images, old deployment archives and development history are excluded. See `LICENSE` for the retained license notice.
