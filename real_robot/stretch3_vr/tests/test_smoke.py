from stretch3_quest_teleop.smoke import run_smoke


def test_scripted_smoke():
    result = run_smoke()
    assert result["ok"] is True
    assert result["final_mode"] == "PRECISION"



def test_mock_backend_fills_both_panel_slots(tmp_path):
    from stretch3_quest_teleop.backends.mock import MockRobotBackend

    (tmp_path / "head").mkdir()
    (tmp_path / "gripper").mkdir()
    (tmp_path / "head" / "a.jpg").write_bytes(b"head-frame")
    (tmp_path / "gripper" / "a.jpg").write_bytes(b"grip-frame")

    # Both panels stream at once now, so the mode no longer selects a camera.
    frames = MockRobotBackend(tmp_path).camera_frames()
    assert frames == {"head": b"head-frame", "gripper": b"grip-frame"}

    assert MockRobotBackend().camera_frames() == {}


def test_navigation_images_stand_in_for_the_head_panel(tmp_path):
    from stretch3_quest_teleop.backends.mock import MockRobotBackend

    (tmp_path / "navigation").mkdir()
    (tmp_path / "navigation" / "a.jpg").write_bytes(b"nav-frame")
    frames = MockRobotBackend(tmp_path).camera_frames()
    assert frames == {"head": b"nav-frame"}


def test_flat_camera_directory_feeds_both_slots(tmp_path):
    from stretch3_quest_teleop.backends.mock import MockRobotBackend

    (tmp_path / "card.jpg").write_bytes(b"one-card")
    frames = MockRobotBackend(tmp_path).camera_frames()
    assert frames == {"head": b"one-card", "gripper": b"one-card"}


def test_missing_camera_directory_is_rejected(tmp_path):
    import pytest

    from stretch3_quest_teleop.backends.mock import MockRobotBackend

    with pytest.raises(ValueError):
        MockRobotBackend(tmp_path / "nope")
    with pytest.raises(ValueError):
        MockRobotBackend(tmp_path)


def test_camera_frame_header_identifies_the_panel():
    import pytest

    from stretch3_quest_teleop.models import encode_camera_frame

    assert encode_camera_frame("head", b"jpeg") == b"S3\x01\x00jpeg"
    assert encode_camera_frame("gripper", b"jpeg") == b"S3\x01\x01jpeg"
    with pytest.raises(ValueError):
        encode_camera_frame("navigation", b"jpeg")
