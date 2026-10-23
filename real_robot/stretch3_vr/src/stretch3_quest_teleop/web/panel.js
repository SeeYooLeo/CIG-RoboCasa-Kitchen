// Panel placement geometry. Deliberately free of DOM and WebGL so that
// tests/panel_geometry.mjs can check the projection without a headset.

export const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
export const wrapPi = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));

// Keeps the panel off the floor and off the ceiling however far the operator
// tips their head.
export const PITCH_LIMITS = [-0.7, 0.5];
// Free head translation before the panel follows, so leaning still gives real
// parallax instead of the panel feeling glued to the eyes.
export const POSITION_DEADZONE_M = 0.3;

export function quaternionForward(q) {
  const [x, y, z, w] = q;
  return [
    -2 * (x * z + w * y),
    -2 * (y * z - w * x),
    -(1 - 2 * (x * x + y * y)),
  ];
}

export function headYawPitch(orientation) {
  const forward = quaternionForward(orientation);
  return [
    Math.atan2(-forward[0], -forward[2]),
    Math.asin(clamp(forward[1], -1, 1)),
  ];
}

// out = a * b, column-major to match WebXR's matrices.
export function mat4Multiply(a, b, out) {
  for (let column = 0; column < 4; column += 1) {
    const b0 = b[column * 4];
    const b1 = b[column * 4 + 1];
    const b2 = b[column * 4 + 2];
    const b3 = b[column * 4 + 3];
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2 + a[12 + row] * b3;
    }
  }
  return out;
}

export function createPanel() {
  return { yaw: 0, pitch: 0, anchor: [0, 1.6, 0], aspect: 16 / 9, placed: false };
}

export function placePanel(panel, position, yaw, pitch) {
  panel.yaw = wrapPi(yaw);
  panel.pitch = clamp(pitch, PITCH_LIMITS[0], PITCH_LIMITS[1]);
  panel.anchor = [position[0], position[1], position[2]];
  panel.placed = true;
  return panel;
}

// Lazy body-lock. The panel is frozen while the head stays inside a deadzone
// and then eases along, staying at the deadzone edge. A rigid head-lock is
// what made the full-FOV blit uncomfortable; a world-lock would slide the
// panel out of sight, because turning the head is also what pans the robot
// head in NAV and ALIGN.
export function updatePanel(panel, view, position, orientation, dt) {
  const [headYaw, headPitch] = headYawPitch(orientation);
  if (!panel.placed) return placePanel(panel, position, headYaw, headPitch);
  // World-locked by default. The operator pans the robot head by turning their
  // own head, so a panel that chases the head would never let the picture
  // settle in front of them. "lazy" restores the follow behaviour.
  if (view.follow_mode !== "lazy") return panel;

  const smooth = 1 - Math.exp(-dt / Math.max(0.05, view.follow_time_constant_s));
  const deadzone = (view.follow_deadzone_deg * Math.PI) / 180;

  const yawError = wrapPi(headYaw - panel.yaw);
  if (Math.abs(yawError) > deadzone) {
    panel.yaw = wrapPi(
      panel.yaw + (yawError - Math.sign(yawError) * deadzone) * smooth,
    );
  }
  const pitchError = headPitch - panel.pitch;
  if (Math.abs(pitchError) > deadzone) {
    panel.pitch = clamp(
      panel.pitch + (pitchError - Math.sign(pitchError) * deadzone) * smooth,
      PITCH_LIMITS[0],
      PITCH_LIMITS[1],
    );
  }
  for (let axis = 0; axis < 3; axis += 1) {
    const error = position[axis] - panel.anchor[axis];
    if (Math.abs(error) > POSITION_DEADZONE_M) {
      panel.anchor[axis] +=
        (error - Math.sign(error) * POSITION_DEADZONE_M) * smooth;
    }
  }
  return panel;
}

// Places the quad `distance` metres along the panel direction and turns its
// +Z face back toward the operator.
export function panelModelMatrix(panel, distance, out) {
  const cy = Math.cos(panel.yaw);
  const sy = Math.sin(panel.yaw);
  const cp = Math.cos(panel.pitch);
  const sp = Math.sin(panel.pitch);
  // Ry(yaw) * Rx(pitch), column-major.
  out[0] = cy; out[1] = 0; out[2] = -sy; out[3] = 0;
  out[4] = sy * sp; out[5] = cp; out[6] = cy * sp; out[7] = 0;
  out[8] = sy * cp; out[9] = -sp; out[10] = cy * cp; out[11] = 0;
  out[12] = panel.anchor[0] - sy * cp * distance;
  out[13] = panel.anchor[1] + sp * distance;
  out[14] = panel.anchor[2] - cy * cp * distance;
  out[15] = 1;
  return out;
}

// The compositor-layer path needs the same placement as a position plus a
// quaternion instead of a matrix. Both are checked against panelModelMatrix in
// tests/panel_geometry.mjs so the two paths cannot drift apart.
export function panelPosition(panel, distance) {
  const cp = Math.cos(panel.pitch);
  return [
    panel.anchor[0] - Math.sin(panel.yaw) * cp * distance,
    panel.anchor[1] + Math.sin(panel.pitch) * distance,
    panel.anchor[2] - Math.cos(panel.yaw) * cp * distance,
  ];
}

// Ry(yaw) * Rx(pitch) as an (x, y, z, w) quaternion.
export function panelOrientation(panel) {
  const cy = Math.cos(panel.yaw / 2);
  const sy = Math.sin(panel.yaw / 2);
  const cp = Math.cos(panel.pitch / 2);
  const sp = Math.sin(panel.pitch / 2);
  return [cy * sp, sy * cp, -sy * sp, cy * cp];
}

// Half-extents, in NDC, of the camera image inside a quad-layer texture whose
// aspect is layerAspect. The remaining margin becomes the mode border. Both
// branches preserve contentAspect, so the image is never stretched.
export function layerContentExtent(borderRatio, contentAspect, layerAspect) {
  const bx = clamp(borderRatio, 0.005, 0.2);
  const by = Math.min(0.25, bx * layerAspect);
  let hy = 1 - by;
  let hx = (hy * contentAspect) / layerAspect;
  if (hx > 1 - bx) {
    hx = 1 - bx;
    hy = (hx * layerAspect) / contentAspect;
  }
  return [hx, hy];
}

// Fits an image of the given aspect inside a box, preserving aspect. Sizing by
// width alone would blow up on the Stretch head camera, which is recorded
// rotated: 480x640 at width_m 1.7 would be a 2.27 m tall panel.
export function fitContent(boxWidth, boxHeight, aspect) {
  const safeAspect = Math.max(0.01, aspect);
  let width = boxWidth;
  let height = width / safeAspect;
  if (height > boxHeight) {
    height = boxHeight;
    width = height * safeAspect;
  }
  return [width, height];
}

// Clear space between the panels, converted from the configured angle to
// metres at the panel distance.
export function gapMetres(gapRadians, distance) {
  return 2 * Math.max(0.05, distance) * Math.tan(Math.max(0, gapRadians) / 2);
}

// Sideways position of each panel inside the shared plane, measured from the
// centre line. The pair's outer edges come out symmetric even though the two
// cameras have very different widths.
export function pairLateralOffsets(leftWidth, rightWidth, gap) {
  const total = leftWidth + gap + rightWidth;
  return [leftWidth / 2 - total / 2, total / 2 - rightWidth / 2];
}

// Centres the panel the current mode is actually flown by, and parks the other
// one beside it. Two equal panels side by side leave the eyes picking between
// them every second; giving the active camera the centre line lets the operator
// settle on one picture and keep the other as peripheral awareness.
export function focusLateralOffsets(focusWidth, sideWidth, gap, sideSign) {
  const side = sideSign >= 0 ? 1 : -1;
  return [0, side * (focusWidth / 2 + gap + sideWidth / 2)];
}

// Slides a panel sideways within the plane, leaving yaw and pitch alone so the
// pair stays parallel. Turning each panel to face the operator instead would
// toe them in by their own offset angle, and because the head view is a narrow
// portrait and the gripper view is wide, the two angles differ by 8 degrees —
// enough that the left panel visibly leans away.
export function slotPanelLateral(panel, lateral) {
  return {
    yaw: panel.yaw,
    pitch: panel.pitch,
    anchor: [
      panel.anchor[0] + Math.cos(panel.yaw) * lateral,
      panel.anchor[1],
      panel.anchor[2] - Math.sin(panel.yaw) * lateral,
    ],
  };
}
