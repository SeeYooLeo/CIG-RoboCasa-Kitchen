import {
  clamp,
  createPanel,
  fitContent,
  layerContentExtent,
  mat4Multiply,
  panelModelMatrix,
  panelOrientation,
  panelPosition,
  gapMetres,
  focusLateralOffsets,
  pairLateralOffsets,
  slotPanelLateral,
  updatePanel,
} from "./panel.js";

const ui = {
  connection: document.querySelector("#connection"),
  mode: document.querySelector("#mode"),
  base: document.querySelector("#base"),
  latency: document.querySelector("#latency"),
  button: document.querySelector("#enter-vr"),
  hint: document.querySelector("#hint"),
  events: document.querySelector("#events"),
  diagnostics: document.querySelector("#diagnostics"),
  canvas: document.querySelector("#xr"),
};

let socket;
let session;
let referenceSpace;
let viewerSpace;
let recording = { enabled: false, active: false };
let recordPending = false;
let previousRecordClick = false;

function updateRecording(value) {
  if (!value) return;
  recording = value;
  const text = value.error ? `Recording error: ${value.error}` :
    value.active ? `REC · ${value.frames} frames` :
    value.episode ? `${value.complete ? "Saved" : "Incomplete"} · ${value.episode}` :
    value.enabled ? "Ready to record" : "Recording disabled on server";
  document.querySelector("#record-status").textContent = text;
  document.querySelector("#record-start").disabled = !value.enabled || value.active || recordPending;
  for (const name of ["save", "failure", "abort"]) {
    document.querySelector(`#record-${name}`).disabled = !value.active || recordPending;
  }
}

function recordControl(operation, success = null) {
  if (!socket || socket.readyState !== WebSocket.OPEN || recordPending || !recording.enabled) return;
  recordPending = true;
  socket.send(JSON.stringify({
    type: "record_control", operation, success, task: document.querySelector("#task").value,
  }));
  updateRecording(recording);
}

document.querySelector("#record-start").onclick = () => recordControl("start");
document.querySelector("#record-save").onclick = () => recordControl("save", true);
document.querySelector("#record-failure").onclick = () => recordControl("save", false);
document.querySelector("#record-abort").onclick = () => recordControl("abort");
let sequence = 0;
let lastSent = 0;
let lastClientTime = 0;
let gl;
let program;
let uniforms;
let lastRenderTime = 0;
let currentMode = "NAV";
let reportedFirstFrames = false;
let lastCounterReport = 0;
let alerting = true;
let previousStickClick = false;
let xrBinding = null;
let renderPath = "No active session";

// The border carries the phase into the headset, borrowed from Open-Teach's
// coloured-border feedback. Without it the operator cannot see the mode at
// all, because entering the immersive session hides the 2D page.
const MODE_COLOR = {
  NAV: [0.36, 0.44, 0.51, 1],
  ALIGN: [0.96, 0.73, 0.26, 1],
  MANIP: [0.29, 0.6, 0.98, 1],
  PRECISION: [0.34, 0.89, 0.71, 1],
};
const ALERT_COLOR = [0.98, 0.3, 0.3, 1];

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const LAYER_PIXEL_WIDTH = 1024;
const PLACEHOLDER = new Uint8Array([
  8, 17, 15, 255, 25, 78, 67, 255, 25, 78, 67, 255, 8, 17, 15, 255,
]);

// Server binary frames are "S3" + version + slot index + JPEG. The index is
// this array's order, so the two must stay in step with models.CAMERA_SLOTS.
const SLOTS = [
  { name: "head", label: "Main view: navigation or head D435if, selected by mode", sign: -1 },
  { name: "gripper", label: "Wrist D405", sign: 1 },
];

// Defaults; overridden by the server `hello` frame from config/default.json.
const view = {
  distance_m: 2.0,
  width_m: 1.45,
  height_m: 1.15,
  gap_deg: 3.0,
  focus_by_mode: {},
  side_scale: 0.5,
  focus_scale: 1.0,
  follow_mode: "fixed",
  follow_deadzone_deg: 18.0,
  follow_time_constant_s: 0.35,
  border_ratio: 0.035,
  quad_extent_scale: 0.5,
  adjust_in_manip: false,
  min_distance_m: 0.8,
  max_distance_m: 5.0,
  min_width_m: 0.5,
  max_width_m: 3.0,
};

const panel = createPanel();
const modelMatrix = new Float32Array(16);
const viewProjection = new Float32Array(16);
const mvpMatrix = new Float32Array(16);

const gapRadians = () => (view.gap_deg * Math.PI) / 180;

const borderColor = () =>
  (alerting ? ALERT_COLOR : (MODE_COLOR[currentMode] || MODE_COLOR.NAV));

const poseToObject = (pose) => ({
  position: [pose.transform.position.x, pose.transform.position.y, pose.transform.position.z],
  orientation: [
    pose.transform.orientation.x,
    pose.transform.orientation.y,
    pose.transform.orientation.z,
    pose.transform.orientation.w,
  ],
});

const emptyPose = () => ({ position: [0, 0, 0], orientation: [0, 0, 0, 1] });

function controllerState(source, frame) {
  if (!source) {
    return {
      tracked: false, pose: emptyPose(), grip: false, trigger: 0,
      stick_x: 0, stick_y: 0, stick_click: false, primary: false, secondary: false,
    };
  }
  const gripPose = source.gripSpace ? frame.getPose(source.gripSpace, referenceSpace) : null;
  const pad = source.gamepad;
  const buttons = pad?.buttons || [];
  const axes = pad?.axes || [];
  return {
    tracked: Boolean(gripPose),
    pose: gripPose ? poseToObject(gripPose) : emptyPose(),
    trigger: buttons[0]?.value || 0,
    grip: Boolean(buttons[1]?.pressed),
    stick_x: axes.at(-2) || 0,
    stick_y: axes.at(-1) || 0,
    stick_click: Boolean(buttons[3]?.pressed),
    primary: Boolean(buttons[4]?.pressed),
    secondary: Boolean(buttons[5]?.pressed),
  };
}

// Returns the right controller every frame for local view adjustment, but only
// transmits to the robot at the protocol's ~30 Hz.
function sendFrame(time, frame, head) {
  if (!socket || socket.readyState !== WebSocket.OPEN || !referenceSpace) return null;
  let leftSource;
  let rightSource;
  for (const source of session.inputSources) {
    if (source.handedness === "left") leftSource = source;
    if (source.handedness === "right") rightSource = source;
  }
  const right = controllerState(rightSource, frame);
  const left = controllerState(leftSource, frame);
  if (left.stick_click && !previousRecordClick && !left.grip && !right.grip) {
    recordControl(recording.active ? "save" : "start", recording.active ? true : null);
  }
  previousRecordClick = Boolean(left.stick_click);
  if (recordPending) return right;
  if (!head || time - lastSent < 33) return right;
  lastSent = time;
  lastClientTime = performance.now();
  socket.send(JSON.stringify({
    type: "vr_frame",
    protocol: 2,
    sequence: sequence++,
    client_time_ms: lastClientTime,
    hmd: { tracked: true, pose: poseToObject(head) },
    left,
    right,
  }));
  return right;
}

// The robot controller ignores the right stick, so it is free for view
// adjustment. Nothing in here reaches the robot.
function adjustView(right, dt) {
  if (!right) return;
  // Locked during manipulation by default. The right thumb rests against the
  // stick while the grip holds the end effector, and having the picture jump
  // in size in the middle of a grasp is worse than not being able to resize.
  if (
    !view.adjust_in_manip
    && (currentMode === "MANIP" || currentMode === "PRECISION")
  ) {
    previousStickClick = Boolean(right.stick_click);
    return;
  }
  const deadzone = 0.2;
  if (Math.abs(right.stick_y) > deadzone) {
    view.distance_m = clamp(
      view.distance_m - right.stick_y * 1.2 * dt,
      view.min_distance_m,
      view.max_distance_m,
    );
  }
  if (Math.abs(right.stick_x) > deadzone) {
    const width = clamp(
      view.width_m * (1 + right.stick_x * 0.8 * dt),
      view.min_width_m,
      view.max_width_m,
    );
    view.height_m *= width / view.width_m;
    view.width_m = width;
  }
  // Essential with a world-locked panel: this is how the pair comes back in
  // front of the operator after they have turned away.
  if (right.stick_click && !previousStickClick) panel.placed = false;
  previousStickClick = Boolean(right.stick_click);
}

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
  }
  return shader;
}

function makeTexture() {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA, 2, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, PLACEHOLDER,
  );
  return texture;
}

function initGraphics() {
  gl = ui.canvas.getContext("webgl", { xrCompatible: true, alpha: true });
  if (!gl) throw new Error("WebGL unavailable");
  const vertex = compile(gl.VERTEX_SHADER, `
    attribute vec2 p;
    varying vec2 uv;
    uniform mat4 uMvp;
    uniform vec2 uHalfSize;
    void main() {
      uv = vec2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
      gl_Position = uMvp * vec4(p.x * uHalfSize.x, p.y * uHalfSize.y, 0.0, 1.0);
    }`);
  const fragment = compile(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying vec2 uv;
    uniform sampler2D camera;
    uniform vec4 uColor;
    uniform float uUseTexture;
    uniform float uRecording;
    void main() {
      gl_FragColor = mix(uColor, texture2D(camera, uv), uUseTexture);
      if (uUseTexture > 0.5 && uRecording > 0.5 && uv.x < 0.07 && uv.y < 0.07) {
        gl_FragColor = vec4(1.0, 0.05, 0.05, 1.0);
      }
    }`);
  program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "WebGL link failed");
  }
  gl.useProgram(program);
  uniforms = {
    mvp: gl.getUniformLocation(program, "uMvp"),
    halfSize: gl.getUniformLocation(program, "uHalfSize"),
    color: gl.getUniformLocation(program, "uColor"),
    useTexture: gl.getUniformLocation(program, "uUseTexture"),
    recording: gl.getUniformLocation(program, "uRecording"),
  };
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const location = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);

  for (const slot of SLOTS) {
    slot.texture = makeTexture();
    slot.aspect = 4 / 3;
    slot.layerAspect = 4 / 3;
    slot.bitmap = null;
    slot.dirty = false;
    slot.layer = null;
    slot.framebuffer = null;
    slot.checked = false;
    slot.preview = document.querySelector(`#preview-${slot.name}`);
  }
}

function uploadCameras() {
  for (const slot of SLOTS) {
    if (!slot.dirty || !slot.bitmap) continue;
    slot.aspect = slot.bitmap.width / Math.max(1, slot.bitmap.height);
    slot.uploaded = (slot.uploaded || 0) + 1;
    gl.bindTexture(gl.TEXTURE_2D, slot.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, slot.bitmap);
    slot.dirty = false;
  }
}

// Which panel the current mode puts on the centre line, or -1 to keep the pair
// side by side. Driving and manipulating are watched through different cameras,
// so the one that matters follows the mode.
function focusIndex() {
  const name = (view.focus_by_mode || {})[currentMode];
  return name ? SLOTS.findIndex((slot) => slot.name === name) : -1;
}

// Geometry shared by both render paths, so a fallback cannot move the panels.
function layout() {
  const focus = focusIndex();
  const side = clamp(view.side_scale, 0.2, 1.0);
  // The focused panel gets a bigger box, not just the centre line. A portrait
  // camera fits by height, so without this the rotated navigation view lands
  // no wider than the panel it is supposed to be dominating.
  const main = clamp(view.focus_scale, 1.0, 3.0);
  const sizes = SLOTS.map((slot, index) => {
    const box = focus < 0 ? 1 : (index === focus ? main : side);
    return fitContent(view.width_m * box, view.height_m * box, slot.aspect);
  });
  const gap = gapMetres(gapRadians(), view.distance_m);
  let offsets;
  if (focus < 0) {
    offsets = pairLateralOffsets(sizes[0][0], sizes[1][0], gap);
  } else {
    const other = focus === 0 ? 1 : 0;
    const pair = focusLateralOffsets(
      sizes[focus][0], sizes[other][0], gap, SLOTS[other].sign,
    );
    offsets = [];
    offsets[focus] = pair[0];
    offsets[other] = pair[1];
  }
  return SLOTS.map((slot, index) => ({
    slot,
    contentWidth: sizes[index][0],
    contentHeight: sizes[index][1],
    lateral: offsets[index],
    placed: slotPanelLateral(panel, offsets[index]),
  }));
}

// Half the pair's total width, in metres, for the diagnostics readout.
function pairHalfWidth() {
  const placements = layout();
  return Math.max(
    ...placements.map(
      ({ contentWidth, lateral }) => Math.abs(lateral) + contentWidth / 2,
    ),
  );
}

// --- compositor layers -----------------------------------------------------

function createSlotLayer(slot) {
  const previous = slot.layer;
  slot.layerAspect = clamp(slot.aspect, 0.4, 3.0);
  const height = Math.round(clamp(LAYER_PIXEL_WIDTH / slot.layerAspect, 256, 2048));
  slot.layer = xrBinding.createQuadLayer({
    space: referenceSpace,
    viewPixelWidth: LAYER_PIXEL_WIDTH,
    viewPixelHeight: height,
    layout: "mono",
  });
  slot.checked = false;
  if (!slot.framebuffer) slot.framebuffer = gl.createFramebuffer();
  return previous;
}

function setupQuadLayers() {
  if (!window.XRWebGLBinding || !window.XRRigidTransform) return false;
  if (session.enabledFeatures && !session.enabledFeatures.includes("layers")) return false;
  try {
    xrBinding = new XRWebGLBinding(session, gl);
    if (typeof xrBinding.createQuadLayer !== "function") throw new Error("no createQuadLayer");
    for (const slot of SLOTS) createSlotLayer(slot);
    session.updateRenderState({ layers: SLOTS.map((slot) => slot.layer) });
    return true;
  } catch (error) {
    xrBinding = null;
    for (const slot of SLOTS) slot.layer = null;
    return false;
  }
}

function reportToRobot(message) {
  // The Quest browser console cannot be read from the robot, so anything that
  // breaks rendering has to travel back over the control link to reach a log.
  try {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "client_error", message: String(message) }));
    }
  } catch (error) { /* reporting must never break the session */ }
}

function fallbackToProjection(reason) {
  reportToRobot(`fallback to projection: ${reason}`);
  for (const slot of SLOTS) {
    if (slot.layer) {
      try { slot.layer.destroy(); } catch (error) { /* already gone */ }
    }
    slot.layer = null;
    if (slot.framebuffer) {
      gl.deleteFramebuffer(slot.framebuffer);
      slot.framebuffer = null;
    }
  }
  xrBinding = null;
  renderPath = "Projection layer (fallback)";
  try {
    // The render state rejects layers and baseLayer in one call.
    session.updateRenderState({ layers: [] });
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    ui.events.textContent = `Compositor layers unavailable; using projection rendering: ${reason}`;
  } catch (error) {
    ui.events.textContent = `Compositor and fallback rendering failed; exit and re-enter: ${reason}`;
    session.end();
  }
}

function renderQuadLayers(frame) {
  let recreated = false;
  const retired = [];
  for (const slot of SLOTS) {
    if (Math.abs(slot.aspect - slot.layerAspect) > 0.02) {
      const previous = createSlotLayer(slot);
      if (previous) retired.push(previous);
      slot.pendingActivation = true;
      recreated = true;
    }
  }
  if (recreated) {
    session.updateRenderState({ layers: SLOTS.map((slot) => slot.layer) });
    for (const layer of retired) layer.destroy();
  }

  for (const { slot, contentWidth, placed } of layout()) {
    if (slot.pendingActivation) {
      // updateRenderState applies from the next frame, so a layer created above
      // is not in this frame's render state yet and getSubImage on it throws
      // InvalidStateError -- which would drop the whole session into the
      // projection fallback. Skip this one panel for one frame.
      slot.pendingActivation = false;
      continue;
    }
    const [hx, hy] = layerContentExtent(
      view.border_ratio, slot.aspect, slot.layerAspect,
    );
    const layerWidth = contentWidth / hx;
    slot.layer.width = layerWidth * view.quad_extent_scale;
    slot.layer.height = (layerWidth / slot.layerAspect) * view.quad_extent_scale;
    const position = panelPosition(placed, view.distance_m);
    const orientation = panelOrientation(placed);
    slot.layer.transform = new XRRigidTransform(
      { x: position[0], y: position[1], z: position[2] },
      { x: orientation[0], y: orientation[1], z: orientation[2], w: orientation[3] },
    );

    const subImage = xrBinding.getSubImage(slot.layer, frame);
    gl.bindFramebuffer(gl.FRAMEBUFFER, slot.framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, subImage.colorTexture, 0,
    );
    if (!slot.checked) {
      slot.checked = true;
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        throw new Error("layer framebuffer incomplete");
      }
    }
    const viewport = subImage.viewport;
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.disable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(uniforms.mvp, false, IDENTITY);

    gl.uniform1f(uniforms.useTexture, 0);
    gl.uniform4fv(uniforms.color, borderColor());
    gl.uniform2f(uniforms.halfSize, 1, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    slot.drawn = (slot.drawn || 0) + 1;
    gl.bindTexture(gl.TEXTURE_2D, slot.texture);
    gl.uniform1f(uniforms.useTexture, 1);
    gl.uniform2f(uniforms.halfSize, hx, hy);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// Fallback: draw the same two panels as textured quads into the eye buffers.
function renderProjection(viewer) {
  const layer = session.renderState.baseLayer;
  if (!layer) return;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.viewport(0, 0, layer.framebufferWidth, layer.framebufferHeight);
  // Transparent in immersive-ar so passthrough surrounds the panels, black in
  // immersive-vr.
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  gl.disable(gl.DEPTH_TEST);
  const color = borderColor();

  for (const eye of viewer.views) {
    const viewport = layer.getViewport(eye);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    mat4Multiply(eye.projectionMatrix, eye.transform.inverse.matrix, viewProjection);

    for (const { slot, contentWidth, contentHeight, placed } of layout()) {
      panelModelMatrix(placed, view.distance_m, modelMatrix);
      mat4Multiply(viewProjection, modelMatrix, mvpMatrix);
      gl.uniformMatrix4fv(uniforms.mvp, false, mvpMatrix);

      const halfWidth = contentWidth / 2;
      const halfHeight = contentHeight / 2;
      const border = Math.max(0.012, halfWidth * view.border_ratio);

      // Border first, camera image over it; no depth test, so draw order wins
      // and the two coplanar quads cannot z-fight.
      gl.uniform1f(uniforms.useTexture, 0);
      gl.uniform4fv(uniforms.color, color);
      gl.uniform2f(uniforms.halfSize, halfWidth + border, halfHeight + border);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      gl.bindTexture(gl.TEXTURE_2D, slot.texture);
      gl.uniform1f(uniforms.useTexture, 1);
      gl.uniform2f(uniforms.halfSize, halfWidth, halfHeight);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }
}

function render(time, frame) {
  if (!session) return;
  session.requestAnimationFrame(render);
  const dt = lastRenderTime ? Math.min(0.1, (time - lastRenderTime) / 1000) : 0.016;
  lastRenderTime = time;

  // The viewer reference space gives the head pose without depending on a
  // projection layer, so both render paths use the same source.
  const head = viewerSpace ? frame.getPose(viewerSpace, referenceSpace) : null;
  adjustView(sendFrame(time, frame, head), dt);
  uploadCameras();
  gl.useProgram(program);
  gl.uniform1f(uniforms.recording, recording.active && !recording.error ? 1 : 0);
  if (!head) return;

  const position = head.transform.position;
  const orientation = head.transform.orientation;
  updatePanel(
    panel,
    view,
    [position.x, position.y, position.z],
    [orientation.x, orientation.y, orientation.z, orientation.w],
    dt,
  );

  if (SLOTS[0].layer) {
    try {
      renderQuadLayers(frame);
    } catch (error) {
      fallbackToProjection(error.message);
    }
    if (time - lastCounterReport > 5000) {
      lastCounterReport = time;
      reportToRobot(
        `counters mode=${currentMode} path=${renderPath} ` + SLOTS.map(
          (slot) => `${slot.name}[rx=${slot.received || 0} dec=${slot.decoded || 0} `
            + `up=${slot.uploaded || 0} draw=${slot.drawn || 0} `
            + `a=${slot.aspect.toFixed(3)} la=${slot.layerAspect.toFixed(3)}]`,
        ).join(" "),
      );
    }
    if (!reportedFirstFrames && SLOTS.every((slot) => slot.bitmap)) {
      reportedFirstFrames = true;
      reportToRobot(
        "first frames " + SLOTS.map(
          (slot) => `${slot.name}=${slot.bitmap.width}x${slot.bitmap.height}`,
        ).join(" "),
      );
    }
    return;
  }
  const viewer = frame.getViewerPose(referenceSpace);
  if (viewer) renderProjection(viewer);
}

async function enterVR() {
  // Passthrough first: panels floating in the real room is what makes this
  // feel like Quest browser windows rather than a blindfold.
  for (const mode of ["immersive-ar", "immersive-vr"]) {
    try {
      if (navigator.xr.isSessionSupported) {
        if (!(await navigator.xr.isSessionSupported(mode))) continue;
      }
      session = await navigator.xr.requestSession(mode, {
        requiredFeatures: ["local-floor"],
        optionalFeatures: ["layers"],
      });
      break;
    } catch (error) {
      session = null;
    }
  }
  if (!session) {
    ui.events.textContent = "Cannot enter an immersive session: neither immersive-ar nor immersive-vr is available.";
    return;
  }
  try {
    await gl.makeXRCompatible();
    referenceSpace = await session.requestReferenceSpace("local-floor");
    viewerSpace = await session.requestReferenceSpace("viewer");
    if (setupQuadLayers()) {
      renderPath = "Compositor XRQuadLayer";
    } else {
      renderPath = "Projection layer";
      session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    }
    ui.events.textContent = `Rendering path: ${renderPath}`;
    panel.placed = false;
    lastRenderTime = 0;
    session.addEventListener("end", () => {
      session = null;
      referenceSpace = null;
      viewerSpace = null;
      xrBinding = null;
      for (const slot of SLOTS) {
        slot.layer = null;
        slot.framebuffer = null;
      }
      panel.placed = false;
      ui.button.textContent = "Re-enter VR control";
      renderDiagnostics();
    });
    session.requestAnimationFrame(render);
  } catch (error) {
    ui.events.textContent = `Cannot enter VR: ${error.message}`;
    try { session.end(); } catch (ignored) { /* already ending */ }
    session = null;
  }
}

function handleCameraFrame(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 5 || bytes[0] !== 0x53 || bytes[1] !== 0x33) return;
  const slot = SLOTS[bytes[3]];
  if (!slot) return;
  slot.received = (slot.received || 0) + 1;
  const jpeg = new Blob([bytes.subarray(4)], { type: "image/jpeg" });
  createImageBitmap(jpeg).then((bitmap) => {
    slot.decoded = (slot.decoded || 0) + 1;
    if (slot.bitmap) slot.bitmap.close();
    slot.bitmap = bitmap;
    slot.dirty = true;
    if (!session && slot.preview) {
      const url = URL.createObjectURL(jpeg);
      slot.preview.src = url;
      if (slot.previewUrl) URL.revokeObjectURL(slot.previewUrl);
      slot.previewUrl = url;
    }
  }).catch((error) => {
    // A corrupt frame is not worth tearing the link down, but a decode that
    // fails every time looks exactly like a frozen panel, so say so once.
    if (!slot.reportedDecodeFailure) {
      slot.reportedDecodeFailure = true;
      reportToRobot(`decode failed on ${slot.name} (${jpeg.size} bytes): ${error}`);
    }
  });
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.addEventListener("open", () => {
    ui.connection.textContent = "Control link connected";
    ui.connection.className = "status ok";
    ui.button.disabled = false;
  });
  socket.addEventListener("close", () => {
    ui.connection.textContent = "Connection lost; the robot stop watchdog will activate";
    ui.connection.className = "status bad";
    ui.button.disabled = true;
    alerting = true;
    recordPending = false;
    updateRecording({ enabled: false, active: false });
    setTimeout(connect, 1500);
  });
  socket.addEventListener("error", () => {
    ui.connection.textContent = "Cannot connect to the control server";
    ui.connection.className = "status bad";
    alerting = true;
  });
  socket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") {
      handleCameraFrame(event.data);
      return;
    }
    const status = JSON.parse(event.data);
    if (status.type === "hello") {
      updateRecording(status.recording);
      for (const [key, value] of Object.entries(status.view || {})) {
        if (key in view && typeof value === typeof view[key]) view[key] = value;
      }
      renderDiagnostics();
    } else if (status.type === "status") {
      updateRecording(status.recording);
      currentMode = status.mode || currentMode;
      alerting = Boolean(status.stale_input);
      ui.mode.textContent = status.mode;
      ui.base.textContent = status.base.locked ? "Locked" : "Enabled";
      ui.latency.textContent = `${Math.max(0, performance.now() - lastClientTime).toFixed(0)} ms`;
      if (status.events && status.events.length) {
        ui.events.textContent = status.events.join(" · ");
      }
    } else if (status.type === "recording") {
      recordPending = false;
      updateRecording(status);
    } else if (status.type === "error") {
      recordPending = false;
      updateRecording(recording);
      ui.events.textContent = status.message;
    }
  });
}

// Shown on the 2D page, which is the only place the operator can read it
// before entering the session. Without this, "enter VR does nothing" gives no
// clue whether the problem is the certificate, the browser or the headset.
function addDiagnostic(label, value, bad) {
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = value;
  if (bad) detail.className = "bad";
  ui.diagnostics.append(term, detail);
}

function describeLayout() {
  const widths = layout().map(({ contentWidth }) => contentWidth.toFixed(2));
  const total = (2 * Math.atan(pairHalfWidth() / view.distance_m) * 180) / Math.PI;
  return `${widths.join(" m + ")} m wide, total ${total.toFixed(0)} degrees horizontal field of view`;
}

async function renderDiagnostics() {
  ui.diagnostics.textContent = "";
  addDiagnostic("Page address", location.origin);
  addDiagnostic(
    "Secure context",
    window.isSecureContext ? "Yes" : "No: the browser will reject WebXR",
    !window.isSecureContext,
  );
  if (!navigator.xr) {
    addDiagnostic("WebXR", "Unavailable; open this page in Quest Browser", true);
    return;
  }
  addDiagnostic("WebXR", "Available");
  const [ar, vr] = await Promise.all([
    navigator.xr.isSessionSupported("immersive-ar").catch(() => false),
    navigator.xr.isSessionSupported("immersive-vr").catch(() => false),
  ]);
  addDiagnostic("Passthrough immersive-ar", ar ? "Supported" : "Unsupported; using a black background", !ar);
  addDiagnostic("immersive-vr", vr ? "Supported" : "Unsupported", !vr);
  addDiagnostic(
    "Compositor WebXR Layers",
    window.XRWebGLBinding ? "Available in this browser" : "Unavailable; using projection rendering",
    !window.XRWebGLBinding,
  );
  addDiagnostic("Last rendering path", renderPath);
  addDiagnostic("Camera panels", view.follow_mode === "lazy" ? "Follow head" : "World locked");
  addDiagnostic(
    "Right stick panel adjustment",
    view.adjust_in_manip ? "All modes" : "NAV / ALIGN only; locked during manipulation",
  );
  addDiagnostic("Combined panel field of view", describeLayout());
  if (!ar && !vr) ui.events.textContent = "This device does not support immersive sessions.";
  else if (!ar) ui.events.textContent = "Passthrough is unavailable; camera panels will appear on a black background.";
}

try {
  initGraphics();
  if (!window.isSecureContext) {
    ui.hint.textContent =
      "WebXR requires a secure context. Use HTTPS or http://localhost through adb reverse.";
  }
  if (!navigator.xr) {
    ui.events.textContent = "This browser does not support WebXR; use Quest Browser.";
  }
  renderDiagnostics();
  ui.button.addEventListener("click", enterVR);
  connect();
} catch (error) {
  ui.connection.textContent = error.message;
  ui.connection.className = "status bad";
}
