import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from stretch3_quest_teleop.config import TeleopConfig, default_config_path
from stretch3_quest_teleop.depth_overlay import (
    apply_overlay,
    band_mask,
    edge_mask,
    reference_mm,
    valid_mask,
)

SETTINGS = {
    "mode": "band",
    "depth_scale_m": 0.001,
    "reference": "fixed",
    "reference_m": 0.182,
    "band_m": 0.015,
    "max_valid_m": 1.0,
    "min_valid_fraction": 0.05,
    "alpha": 1.0,
    "band_color_bgr": [0, 255, 255],
}


def scene(fill_mm=182):
    return np.full((32, 48), fill_mm, dtype=np.uint16)


def grey():
    return np.full((32, 48, 3), 100, dtype=np.uint8)


def test_a_dropout_is_not_a_surface_at_zero_metres():
    depth = scene()
    depth[:8, :] = 0
    valid = valid_mask(depth, 1000)
    assert not valid[:8, :].any()
    assert valid[8:, :].all()


def test_the_band_lights_up_the_fingertip_plane_only():
    depth = scene(182)
    depth[0, 0] = 220  # 3.8 cm beyond the band
    depth[0, 1] = 190  # 8 mm inside it
    valid = valid_mask(depth, 1000)
    mask = band_mask(depth, valid, 182.0, 15, np)
    assert not mask[0, 0]
    assert mask[0, 1]
    assert mask[5, 5]


def test_holes_do_not_get_outlined_as_objects():
    """The step from a valid reading to a 0 is a false gradient, not an edge."""
    flat = scene()
    flat[10:20, 10:20] = 0
    valid = valid_mask(flat, 1000)
    assert not edge_mask(flat, valid, 10, np, cv2).any()

    step = scene()
    step[:, 24:] = 300  # a real 11.8 cm depth discontinuity
    valid = valid_mask(step, 1000)
    assert edge_mask(step, valid, 10, np, cv2).any()


def test_the_overlay_disappears_instead_of_showing_an_empty_band():
    depth = np.zeros((32, 48), dtype=np.uint16)
    image = grey()
    assert apply_overlay(image, depth, SETTINGS, np, cv2) is False
    assert (image == 100).all()


def test_it_paints_the_configured_colour_where_the_band_hits():
    image = grey()
    assert apply_overlay(image, scene(182), SETTINGS, np, cv2) is True
    assert (image == np.array([0, 255, 255], dtype=np.uint8)).all()


def test_a_partly_transparent_overlay_leaves_the_scene_visible():
    image = grey()
    apply_overlay(image, scene(182), {**SETTINGS, "alpha": 0.5}, np, cv2)
    assert 100 < int(image[0, 0, 1]) < 255


def test_misaligned_depth_is_refused_rather_than_silently_cropped():
    with pytest.raises(ValueError, match="aligned_depth_to_color"):
        apply_overlay(grey(), np.zeros((16, 16), dtype=np.uint16), SETTINGS, np, cv2)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="depth_overlay.mode"):
        apply_overlay(grey(), scene(), {**SETTINGS, "mode": "peak"}, np, cv2)


def test_center_roi_ignores_dropouts_when_taking_the_median():
    depth = scene(250)
    depth[12:16, 18:30] = 0  # half the ROI drops out
    assert reference_mm(depth, {**SETTINGS, "reference": "center_roi"}, np) == 250.0


def test_center_roi_reports_nothing_rather_than_guessing():
    depth = scene(250)
    depth[12:20, 18:30] = 0  # the whole ROI drops out
    assert reference_mm(depth, {**SETTINGS, "reference": "center_roi"}, np) is None


def test_the_shipped_configuration_draws_no_overlay():
    overlays = TeleopConfig.load(default_config_path()).raw["depth_overlay"]
    assert not any(entry["enabled"] for entry in overlays.values())


def test_the_cue_fades_in_before_the_band_instead_of_flashing_on():
    """Blank until 1.5 cm away is useless; the operator needs a gradient."""
    from stretch3_quest_teleop.depth_overlay import band_weight

    depth = np.array([[182, 190, 250, 300, 400]], dtype=np.uint16)
    valid = valid_mask(depth, 1000)
    graded = band_weight(depth, valid, 182.0, 15, 120, np)
    assert graded[0, 0] == 1.0                     # on the fingertip plane
    assert graded[0, 1] == 1.0                     # 8 mm off, still inside it
    assert 0.0 < graded[0, 3] < graded[0, 2] < 1.0  # fades with distance
    assert graded[0, 4] == 0.0                     # past the capture window

    hard = band_weight(depth, valid, 182.0, 15, 0, np)
    assert list(hard[0]) == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_the_graded_cue_still_ignores_dropouts():
    from stretch3_quest_teleop.depth_overlay import band_weight

    depth = np.array([[0, 190]], dtype=np.uint16)
    valid = valid_mask(depth, 1000)
    assert band_weight(depth, valid, 182.0, 15, 120, np)[0, 0] == 0.0


def test_a_distant_surface_is_tinted_faintly_not_painted_solid():
    image = grey()
    settings = {**SETTINGS, "falloff_m": 0.12, "alpha": 0.55}
    apply_overlay(image, scene(280), settings, np, cv2)  # 10 cm short of the plane
    assert not (image == 100).all()          # something is drawn
    assert abs(int(image[0, 0, 1]) - 100) < 60  # but faintly


def test_gamma_keeps_the_far_field_faint_without_dimming_the_grasp_plane():
    """A linear ramp tints half a cluttered panel at middling strength."""
    from stretch3_quest_teleop.depth_overlay import band_weight

    depth = np.array([[182, 288]], dtype=np.uint16)  # on plane, and 10 cm short
    valid = valid_mask(depth, 1000)
    linear = band_weight(depth, valid, 182.0, 15, 120, np, 1.0)
    shaped = band_weight(depth, valid, 182.0, 15, 120, np, 2.5)
    assert shaped[0, 0] == linear[0, 0] == 1.0  # the cue itself is untouched
    assert shaped[0, 1] < linear[0, 1] / 3.0    # the far field falls away
