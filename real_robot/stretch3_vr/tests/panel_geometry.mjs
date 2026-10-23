// Geometry checks for the VR camera panel. The headset cannot be scripted, so
// these verify the maths that decides where the panel lands and how large it
// looks. Run with node tests/panel_geometry.mjs.
import assert from "node:assert/strict";

import {
  createPanel,
  fitContent,
  focusLateralOffsets,
  gapMetres,
  headYawPitch,
  layerContentExtent,
  mat4Multiply,
  pairLateralOffsets,
  panelModelMatrix,
  panelOrientation,
  panelPosition,
  placePanel,
  slotPanelLateral,
  updatePanel,
} from "../src/stretch3_quest_teleop/web/panel.js";

const DEGREES = Math.PI / 180;
const close = (actual, expected, tolerance, label) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, got ${actual}`,
  );

// The follow tests below exercise the optional lazy mode; the shipped default
// is the world lock, checked in the side-by-side block at the end.
const view = {
  distance_m: 2.0,
  width_m: 1.7,
  height_m: 1.15,
  follow_mode: "lazy",
  follow_deadzone_deg: 18.0,
  follow_time_constant_s: 0.35,
};

function perspective(fovYRadians, aspect, near, far) {
  const f = 1 / Math.tan(fovYRadians / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

// Head at eye height looking down -Z; the view matrix is the inverse.
function viewMatrixAtEyeHeight(height) {
  const out = new Float32Array(16);
  out[0] = 1; out[5] = 1; out[10] = 1; out[15] = 1;
  out[13] = -height;
  return out;
}

function project(model, viewMatrix, projection, point) {
  const mvp = new Float32Array(16);
  const viewProjection = new Float32Array(16);
  mat4Multiply(projection, viewMatrix, viewProjection);
  mat4Multiply(viewProjection, model, mvp);
  const [x, y, z] = point;
  const clip = [0, 0, 0, 0];
  for (let row = 0; row < 4; row += 1) {
    clip[row] = mvp[row] * x + mvp[4 + row] * y + mvp[8 + row] * z + mvp[12 + row];
  }
  return [clip[0] / clip[3], clip[1] / clip[3], clip[3]];
}

// --- matrix helper ---------------------------------------------------------

{
  const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const translate = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 5, 1]);
  const out = new Float32Array(16);
  mat4Multiply(identity, translate, out);
  assert.deepEqual(Array.from(out), Array.from(translate), "identity * T === T");

  // Column-major convention check: T(3,4,5) applied to the origin.
  const scale = new Float32Array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
  mat4Multiply(scale, translate, out);
  close(out[12], 6, 1e-9, "scale then translate x");
  close(out[13], 8, 1e-9, "scale then translate y");
}

// --- head orientation ------------------------------------------------------

{
  const yaw = 30 * DEGREES;
  const quaternion = [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
  const [measuredYaw, measuredPitch] = headYawPitch(quaternion);
  close(measuredYaw, yaw, 1e-6, "head yaw round-trip");
  close(measuredPitch, 0, 1e-6, "head pitch round-trip");
}

// --- placement -------------------------------------------------------------

{
  const panel = placePanel(createPanel(), [0, 1.6, 0], 0, 0);
  const model = panelModelMatrix(panel, view.distance_m, new Float32Array(16));
  close(model[12], 0, 1e-6, "panel dead ahead x");
  close(model[13], 1.6, 1e-6, "panel dead ahead y");
  close(model[14], -2.0, 1e-6, "panel dead ahead z");
}

{
  const panel = placePanel(createPanel(), [0, 1.6, 0], 30 * DEGREES, 0);
  const model = panelModelMatrix(panel, view.distance_m, new Float32Array(16));
  close(model[12], -1.0, 1e-6, "panel yawed x");
  close(model[14], -Math.sqrt(3), 1e-6, "panel yawed z");

  // The +Z face must point back at the operator, otherwise the camera image
  // would be behind the quad.
  const normal = [model[8], model[9], model[10]];
  const toOperator = [
    panel.anchor[0] - model[12],
    panel.anchor[1] - model[13],
    panel.anchor[2] - model[14],
  ];
  const length = Math.hypot(...toOperator);
  const dot =
    (normal[0] * toOperator[0] + normal[1] * toOperator[1] + normal[2] * toOperator[2]) /
    length;
  close(dot, 1.0, 1e-6, "panel faces the operator");
}

// --- apparent size ---------------------------------------------------------

{
  const panel = placePanel(createPanel(), [0, 1.6, 0], 0, 0);
  const model = panelModelMatrix(panel, view.distance_m, new Float32Array(16));
  const projection = perspective(90 * DEGREES, 1, 0.1, 100);
  const viewMatrix = viewMatrixAtEyeHeight(1.6);

  const halfWidth = view.width_m / 2;
  const halfHeight = halfWidth / panel.aspect;

  const [centreX, centreY, w] = project(model, viewMatrix, projection, [0, 0, 0]);
  close(centreX, 0, 1e-6, "panel centre is straight ahead");
  close(centreY, 0, 1e-6, "panel centre is at eye height");
  close(w, view.distance_m, 1e-6, "panel sits at the configured depth");

  const [cornerX, cornerY] = project(model, viewMatrix, projection, [halfWidth, halfHeight, 0]);
  close(cornerX, 0.425, 1e-4, "panel corner NDC x");
  close(cornerY, 0.2390625, 1e-4, "panel corner NDC y");

  // The whole point of the change: a bounded window instead of the full FOV.
  const horizontalFov = 2 * Math.atan(halfWidth / view.distance_m) / DEGREES;
  close(horizontalFov, 46.05, 0.05, "horizontal field of view");
  assert.ok(horizontalFov < 60, "panel must stay well inside the Quest 3S FOV");
}

// --- lazy follow -----------------------------------------------------------

{
  const yawQuaternion = (yaw) => [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
  const panel = placePanel(createPanel(), [0, 1.6, 0], 0, 0);

  updatePanel(panel, view, [0, 1.6, 0], yawQuaternion(10 * DEGREES), 0.1);
  close(panel.yaw, 0, 1e-9, "head movement inside the deadzone leaves the panel put");

  updatePanel(panel, view, [0, 1.6, 0], yawQuaternion(40 * DEGREES), 0.1);
  close(panel.yaw, 0.095428, 1e-5, "panel eases toward a large head turn");
  assert.ok(panel.yaw < 40 * DEGREES, "panel must lag the head, never lead it");

  for (let step = 0; step < 200; step += 1) {
    updatePanel(panel, view, [0, 1.6, 0], yawQuaternion(40 * DEGREES), 0.1);
  }
  close(
    panel.yaw,
    (40 - 18) * DEGREES,
    1e-4,
    "panel settles exactly one deadzone behind the head",
  );
}

{
  // Head translation inside the deadzone must not drag the panel, so leaning
  // in still produces parallax.
  const panel = placePanel(createPanel(), [0, 1.6, 0], 0, 0);
  updatePanel(panel, view, [0.2, 1.6, 0], [0, 0, 0, 1], 0.1);
  close(panel.anchor[0], 0, 1e-9, "small lean does not move the panel");

  for (let step = 0; step < 400; step += 1) {
    updatePanel(panel, view, [1.0, 1.6, 0], [0, 0, 0, 1], 0.1);
  }
  close(panel.anchor[0], 0.7, 1e-4, "large translation drags the panel along");
}

// --- compositor-layer placement --------------------------------------------

// The XRQuadLayer path sets a position and a quaternion instead of a matrix.
// If these ever disagree with panelModelMatrix the panel would jump whenever
// the headset falls back from a quad layer to projection rendering.
function quaternionToMatrix([x, y, z, w]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

const placements = [
  [0, 0],
  [30 * DEGREES, 0],
  [0, -25 * DEGREES],
  [-110 * DEGREES, 18 * DEGREES],
];

for (const [yaw, pitch] of placements) {
  const panel = placePanel(createPanel(), [0.3, 1.55, -0.2], yaw, pitch);
  const model = panelModelMatrix(panel, view.distance_m, new Float32Array(16));

  const position = panelPosition(panel, view.distance_m);
  for (let axis = 0; axis < 3; axis += 1) {
    close(position[axis], model[12 + axis], 1e-6, `quad position axis ${axis} yaw ${yaw}`);
  }

  const quaternion = panelOrientation(panel);
  close(Math.hypot(...quaternion), 1, 1e-9, "quad quaternion is normalised");

  const rotation = quaternionToMatrix(quaternion);
  for (let column = 0; column < 3; column += 1) {
    for (let row = 0; row < 3; row += 1) {
      close(
        rotation[row][column],
        model[column * 4 + row],
        1e-6,
        `quad rotation [${row}][${column}] yaw ${yaw}`,
      );
    }
  }
}

// --- quad layer texture layout ---------------------------------------------

// The camera image must come out exactly view.width_m wide and unstretched,
// whatever the camera aspect, with a visible border margin left over.
for (const contentAspect of [16 / 9, 4 / 3, 1.0, 640 / 480]) {
  for (const layerAspect of [16 / 9, 4 / 3]) {
    const [hx, hy] = layerContentExtent(0.035, contentAspect, layerAspect);
    assert.ok(hx > 0 && hx <= 1, `hx in range for ${contentAspect}/${layerAspect}`);
    assert.ok(hy > 0 && hy <= 1, `hy in range for ${contentAspect}/${layerAspect}`);
    assert.ok(hx < 1 && hy < 1, "a border margin must remain on both axes");

    // app.js sizes the layer as layerWidth = width_m / hx.
    const contentWidth = 1.7;
    const layerWidth = contentWidth / hx;
    const layerHeight = layerWidth / layerAspect;
    close(hx * layerWidth, contentWidth, 1e-9, "content is exactly the configured width");
    close(
      (hx * layerWidth) / (hy * layerHeight),
      contentAspect,
      1e-9,
      `content aspect preserved for ${contentAspect} in ${layerAspect}`,
    );
  }
}

// --- box fit for the real Stretch camera aspects ---------------------------

{
  // 16:9 is width-limited and keeps the original 1.7 m panel.
  const [w169, h169] = fitContent(view.width_m, view.height_m, 16 / 9);
  close(w169, 1.7, 1e-9, "16:9 stays width-limited");
  close(h169, 1.7 / (16 / 9), 1e-9, "16:9 height follows from width");

  // The real Stretch cameras are 640x480 and a rotated 480x640.
  for (const [label, aspect] of [["gripper D405", 640 / 480], ["head D435if", 480 / 640]]) {
    const [width, height] = fitContent(view.width_m, view.height_m, aspect);
    close(width / height, aspect, 1e-9, `${label} aspect preserved`);
    assert.ok(width <= view.width_m + 1e-9, `${label} fits the box width`);
    assert.ok(height <= view.height_m + 1e-9, `${label} fits the box height`);

    // Sizing by width alone would have made the portrait head camera 2.27 m
    // tall, which is 59 degrees of vertical field of view at 2 m.
    const vertical = 2 * Math.atan(height / 2 / view.distance_m) / DEGREES;
    assert.ok(vertical < 35, `${label} vertical field of view stays comfortable`);
  }

  const [portraitWidth, portraitHeight] = fitContent(view.width_m, view.height_m, 480 / 640);
  close(portraitHeight, view.height_m, 1e-9, "portrait is height-limited");
  close(portraitWidth, view.height_m * (480 / 640), 1e-9, "portrait width follows height");
}

// --- side-by-side pair, with the real camera aspects -----------------------

{
  // The shipped defaults from config/default.json.
  const pair = {
    distance_m: 2.0,
    width_m: 1.45,
    height_m: 1.15,
    gap_deg: 3.0,
    follow_mode: "fixed",
    follow_deadzone_deg: 18.0,
    follow_time_constant_s: 0.35,
  };
  const HEAD = 480 / 640;     // D435if, recorded rotated: portrait
  const GRIPPER = 640 / 480;  // D405

  const [headWidth] = fitContent(pair.width_m, pair.height_m, HEAD);
  const [gripperWidth] = fitContent(pair.width_m, pair.height_m, GRIPPER);
  const gap = gapMetres((pair.gap_deg * Math.PI) / 180, pair.distance_m);
  const [leftLateral, rightLateral] = pairLateralOffsets(headWidth, gripperWidth, gap);

  // Inner edges exactly one gap apart: no overlap, no drifting apart.
  const headInner = leftLateral + headWidth / 2;
  const gripperInner = rightLateral - gripperWidth / 2;
  close(gripperInner - headInner, gap, 1e-12, "gap between the panels");
  assert.ok(headInner < gripperInner, "the panels must not overlap");

  // Outer edges symmetric about straight ahead, even though the head panel is
  // a narrow portrait and the gripper panel is wide.
  const headOuter = leftLateral - headWidth / 2;
  const gripperOuter = rightLateral + gripperWidth / 2;
  close(headOuter + gripperOuter, 0, 1e-12, "pair is centred on straight ahead");

  const totalDegrees = (2 * Math.atan(gripperOuter / pair.distance_m) * 180) / Math.PI;
  close(totalDegrees, 62.29, 0.05, "combined horizontal span");
  assert.ok(totalDegrees > 52 && totalDegrees < 72, "span stays in the intended band");

  for (const [label, aspect] of [["head", HEAD], ["gripper", GRIPPER]]) {
    const [width, height] = fitContent(pair.width_m, pair.height_m, aspect);
    close(width / height, aspect, 1e-9, `${label} panel keeps its aspect`);
  }

  // --- the panels must be parallel ----------------------------------------
  // Turning each panel to face the operator toed them in by their own offset
  // angle. The two angles differ by 8 degrees for these camera aspects, which
  // reads as the left panel leaning away. Sliding them sideways inside one
  // shared plane keeps every rotation identical.
  for (const [yaw, pitch] of [[0, 0], [40 * DEGREES, -12 * DEGREES], [-95 * DEGREES, 20 * DEGREES]]) {
    const base = placePanel(createPanel(), [0.2, 1.6, -0.4], yaw, pitch);
    const left = panelModelMatrix(
      slotPanelLateral(base, leftLateral), pair.distance_m, new Float32Array(16),
    );
    const right = panelModelMatrix(
      slotPanelLateral(base, rightLateral), pair.distance_m, new Float32Array(16),
    );
    for (let column = 0; column < 3; column += 1) {
      for (let row = 0; row < 3; row += 1) {
        close(
          left[column * 4 + row],
          right[column * 4 + row],
          1e-12,
          `panels share rotation [${row}][${column}] at yaw ${yaw}`,
        );
      }
    }
    // Same height, and separated purely along the plane's right vector.
    close(left[13], right[13], 1e-12, "panels sit at the same height");
    const separation = Math.hypot(right[12] - left[12], right[14] - left[14]);
    close(
      separation,
      rightLateral - leftLateral,
      1e-6,  // the model matrix is a Float32Array
      "separation equals the lateral offset difference",
    );
  }

  // --- world lock ---------------------------------------------------------
  // The operator pans the robot head by turning their own head, so the panels
  // must not follow, however far the head turns.
  const yawQuaternion = (yaw) => [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
  const fixed = placePanel(createPanel(), [0, 1.6, 0], 0, 0);
  for (const degrees of [5, 25, 80, -120]) {
    updatePanel(fixed, pair, [0.4, 1.7, -0.3], yawQuaternion(degrees * DEGREES), 0.1);
    close(fixed.yaw, 0, 1e-12, `panel stays put through a ${degrees} degree head turn`);
    close(fixed.pitch, 0, 1e-12, "panel pitch stays put");
    close(fixed.anchor[0], 0, 1e-12, "panel anchor stays put");
  }

  // "lazy" still follows, so the old behaviour remains reachable.
  const lazy = placePanel(createPanel(), [0, 1.6, 0], 0, 0);
  updatePanel(
    lazy, { ...pair, follow_mode: "lazy" }, [0, 1.6, 0], yawQuaternion(40 * DEGREES), 0.1,
  );
  assert.ok(lazy.yaw > 0, "lazy mode still eases toward the head");
}

console.log("panel geometry: all checks passed");

// --- mode-focused layout ---------------------------------------------------
// Driving and manipulating are watched through different cameras, so the active
// one takes the centre line instead of the pair straddling it.
{
  const gap = 0.1;
  const [focus, side] = focusLateralOffsets(1.2, 0.5, gap, 1);
  assert.equal(focus, 0, "the focused panel sits on the centre line");
  assert.ok(side > 0, "sideSign +1 parks the other panel to the right");
  const clear = Math.abs(side) - 0.5 / 2 - 1.2 / 2;
  assert.ok(Math.abs(clear - gap) < 1e-9, "the configured gap is preserved");

  const [, left] = focusLateralOffsets(1.2, 0.5, gap, -1);
  assert.equal(left, -side, "sideSign -1 mirrors it to the left");
}

{
  // A pair layout centres the pair, not either panel; that is the difference.
  const [a, b] = pairLateralOffsets(1.2, 0.5, 0.1);
  assert.ok(a < 0 && b > 0, "the pair straddles the centre line");
  assert.notEqual(a, 0, "neither panel is centred in the pair layout");
  assert.notEqual(b, 0, "neither panel is centred in the pair layout");
}

{
  // Shrinking the side panel must not move the focused one off centre.
  const [focus] = focusLateralOffsets(1.2, 0.2, 0.1, 1);
  assert.equal(focus, 0, "the focused panel stays centred whatever the side size");
}
