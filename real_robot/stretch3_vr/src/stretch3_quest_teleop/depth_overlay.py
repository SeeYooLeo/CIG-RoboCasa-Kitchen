"""Depth-derived overlays painted onto a camera frame before JPEG encoding.

The wrist D405 publishes colour and aligned depth at the same resolution, so
this is a pixel-wise operation with no reprojection. Compositing on the robot
rather than in the headset keeps the wire format unchanged and costs no extra
bandwidth: the panels still receive one JPEG per camera.

The band mode is the useful one. It highlights the surfaces sitting at the
gripper's own fingertip plane, so a lit-up surface means "closing now would
grasp this". Depth from a stereo pair is an estimate, not a measurement, and
periodic textures can make it confidently wrong, so the overlay is drawn
semi-transparent: the operator can always see the scene underneath it.
"""

from __future__ import annotations

from typing import Any

# cv2.Sobel with ksize=3 responds with 8x the per-pixel gradient.
_SOBEL_GAIN = 8.0

VALID_MODES = ("off", "band", "edges", "both")


def _as_int_mm(value: Any, scale_m: float) -> int:
    return int(round(float(value) / scale_m))


def reference_mm(depth: Any, settings: dict[str, Any], np: Any) -> float | None:
    """Depth of the plane the band highlights, in millimetres.

    "fixed" is the fingertip plane measured off TF once, which is what makes
    the overlay mean something mechanical. "center_roi" instead tracks whatever
    is in front of the camera, which is only useful for eyeballing the sensor.
    """
    scale = float(settings.get("depth_scale_m", 0.001))
    mode = str(settings.get("reference", "fixed"))
    if mode == "fixed":
        return float(_as_int_mm(settings.get("reference_m", 0.182), scale))
    if mode != "center_roi":
        raise ValueError(f"unknown depth_overlay.reference {mode!r}")
    fraction = float(settings.get("roi_fraction", 0.25))
    height, width = depth.shape[:2]
    half_h = max(1, int(height * fraction / 2))
    half_w = max(1, int(width * fraction / 2))
    patch = depth[
        height // 2 - half_h : height // 2 + half_h,
        width // 2 - half_w : width // 2 + half_w,
    ]
    max_valid = _as_int_mm(settings.get("max_valid_m", 1.0), scale)
    samples = patch[(patch > 0) & (patch <= max_valid)]
    if samples.size == 0:
        return None
    return float(np.median(samples))


def valid_mask(depth: Any, max_valid_mm: int) -> Any:
    """Stereo depth reports 0 where matching failed; those are not zero metres."""
    return (depth > 0) & (depth <= max_valid_mm)


def band_mask(depth: Any, valid: Any, ref_mm: float, band_mm: int, np: Any) -> Any:
    delta = np.abs(depth.astype(np.int32) - int(round(ref_mm)))
    return valid & (delta <= band_mm)


def band_weight(
    depth: Any,
    valid: Any,
    ref_mm: float,
    band_mm: int,
    falloff_mm: int,
    np: Any,
    gamma: float = 1.0,
) -> Any:
    """Per-pixel strength of the grasp cue, 1.0 at the fingertip plane.

    A hard-edged band tells the operator nothing until they are already within
    a centimetre of the right distance: the panel stays blank, blank, blank,
    then flashes. Real focus peaking always shows something and firms up as you
    approach, so the cue fades in over `falloff_mm` either side of the band.
    A falloff of zero keeps the plain hard band.

    The ramp is not linear. A whole cluttered scene usually sits inside the
    capture window, and a linear ramp tints most of the panel at middling
    strength, which is noise rather than information. `gamma` above 1 keeps the
    far field barely tinted and lets the cue firm up sharply near the plane,
    which is how peaking reads on a camera.
    """
    delta = np.abs(depth.astype(np.int32) - int(round(ref_mm)))
    if falloff_mm <= 0:
        weight = (delta <= band_mm).astype(np.float32)
    else:
        outside = (delta - band_mm).astype(np.float32) / float(falloff_mm)
        weight = 1.0 - np.clip(outside, 0.0, 1.0)
        if gamma != 1.0:
            weight = np.power(weight, max(0.05, float(gamma)))
    weight[~valid] = 0.0
    return weight


def edge_mask(depth: Any, valid: Any, threshold_mm: int, np: Any, cv2: Any) -> Any:
    """Depth discontinuities, i.e. object silhouettes.

    Holes have to be excluded from the neighbourhood as well as the centre
    pixel: the step from a valid reading to a 0 is a huge false gradient that
    would outline every dropout instead of every object.
    """
    values = depth.astype(np.float32)
    values[~valid] = 0.0
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy) / _SOBEL_GAIN
    kernel = np.ones((3, 3), np.uint8)
    solid = cv2.erode(valid.astype(np.uint8), kernel) > 0
    return solid & (gradient >= threshold_mm)


def blend(image: Any, mask: Any, color_bgr: Any, alpha: float, np: Any) -> None:
    """Paint `color` over the masked pixels, in place."""
    if not mask.any():
        return
    alpha = min(1.0, max(0.0, float(alpha)))
    color = np.asarray(color_bgr, dtype=np.float32)
    region = image[mask].astype(np.float32)
    image[mask] = (region * (1.0 - alpha) + color * alpha).astype(image.dtype)


def blend_weighted(
    image: Any,
    weight: Any,
    near_color_bgr: Any,
    band_color_bgr: Any,
    alpha: float,
    np: Any,
) -> None:
    """Paint the graded cue: dim and `near` coloured far out, full `band` on plane.

    Both the opacity and the hue carry the distance, so the cue reads at a
    glance in peripheral vision without the operator having to compare shades.
    """
    mask = weight > 0.0
    if not mask.any():
        return
    alpha = min(1.0, max(0.0, float(alpha)))
    ramp = weight[mask].astype(np.float32)[:, None]
    near = np.asarray(near_color_bgr, dtype=np.float32)
    band = np.asarray(band_color_bgr, dtype=np.float32)
    color = near * (1.0 - ramp) + band * ramp
    region = image[mask].astype(np.float32)
    blended = region * (1.0 - ramp * alpha) + color * (ramp * alpha)
    image[mask] = blended.astype(image.dtype)


def apply_overlay(
    image: Any, depth: Any, settings: dict[str, Any], np: Any, cv2: Any
) -> bool:
    """Draw the configured overlay onto `image` in place. Returns True if drawn.

    Returns False rather than drawing when too much of the frame dropped out.
    A blank overlay must never be readable as "nothing in front of the gripper",
    so when the sensor stops answering the overlay disappears entirely instead
    of showing an empty band.
    """
    mode = str(settings.get("mode", "band"))
    if mode not in VALID_MODES:
        raise ValueError(
            f"depth_overlay.mode must be one of {VALID_MODES}, got {mode!r}"
        )
    if mode == "off":
        return False
    if depth.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"depth {depth.shape[:2]} does not match colour {image.shape[:2]}; "
            "the overlay needs the aligned_depth_to_color stream"
        )
    scale = float(settings.get("depth_scale_m", 0.001))
    max_valid = _as_int_mm(settings.get("max_valid_m", 1.0), scale)
    valid = valid_mask(depth, max_valid)
    fraction = float(valid.sum()) / float(valid.size)
    if fraction < float(settings.get("min_valid_fraction", 0.05)):
        return False
    alpha = float(settings.get("alpha", 0.55))
    drawn = False
    if mode in ("band", "both"):
        reference = reference_mm(depth, settings, np)
        if reference is not None:
            weight = band_weight(
                depth,
                valid,
                reference,
                _as_int_mm(settings.get("band_m", 0.015), scale),
                _as_int_mm(settings.get("falloff_m", 0.0), scale),
                np,
                float(settings.get("falloff_gamma", 1.0)),
            )
            blend_weighted(
                image,
                weight,
                settings.get("near_color_bgr", [255, 96, 0]),
                settings.get("band_color_bgr", [0, 255, 255]),
                alpha,
                np,
            )
            drawn = True
    if mode in ("edges", "both"):
        mask = edge_mask(
            depth,
            valid,
            _as_int_mm(settings.get("edge_threshold_m", 0.01), scale),
            np,
            cv2,
        )
        blend(image, mask, settings.get("edge_color_bgr", [255, 128, 0]), alpha, np)
        drawn = True
    return drawn
