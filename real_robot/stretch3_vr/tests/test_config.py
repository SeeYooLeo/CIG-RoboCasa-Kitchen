import json

from stretch3_quest_teleop.config import TeleopConfig, default_config_path, merge


def test_overlay_replaces_only_the_keys_it_names(tmp_path):
    overlay = tmp_path / "slow.json"
    overlay.write_text(json.dumps({"base": {"nav_linear_mps": 0.05}}))

    plain = TeleopConfig.load(default_config_path())
    slowed = TeleopConfig.load(default_config_path(), [overlay])

    assert slowed.base["nav_linear_mps"] == 0.05
    assert slowed.base["nav_angular_rps"] == plain.base["nav_angular_rps"]
    assert slowed.robot == plain.robot
    assert slowed.overlays == ("slow.json",)
    assert plain.overlays == ()
    # A load must not contaminate the next one: an operator who forgets the
    # overlay flag has to get full speed back, not the previous session's.
    assert TeleopConfig.load(default_config_path()).base == plain.base


def test_overlays_apply_in_order(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"base": {"nav_linear_mps": 0.05}}))
    second.write_text(json.dumps({"base": {"nav_linear_mps": 0.01}}))
    config = TeleopConfig.load(default_config_path(), [first, second])
    assert config.base["nav_linear_mps"] == 0.01
    assert config.overlays == ("a.json", "b.json")


def test_merge_leaves_its_inputs_alone():
    base = {"base": {"a": 1, "b": 2}}
    result = merge(base, {"base": {"b": 3}})
    assert result == {"base": {"a": 1, "b": 3}}
    assert base == {"base": {"a": 1, "b": 2}}


def test_shipped_stage_overlays():
    root = default_config_path().parent
    full = TeleopConfig.load(default_config_path())

    stopped = TeleopConfig.load(
        default_config_path(), [root / "overlay_base_locked.json"]
    )
    for key in (
        "nav_linear_mps",
        "nav_angular_rps",
        "align_linear_mps",
        "align_angular_rps",
        "manip_linear_mps",
        "manip_angular_rps",
        "ik_base_max_mps",
    ):
        assert stopped.base[key] == 0.0, key

    slow = TeleopConfig.load(default_config_path(), [root / "overlay_slow.json"])
    for key in ("nav_linear_mps", "nav_angular_rps", "ik_base_max_mps"):
        assert 0.0 < slow.base[key] < full.base[key], key
