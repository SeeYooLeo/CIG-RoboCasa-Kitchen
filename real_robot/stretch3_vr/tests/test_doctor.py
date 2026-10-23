from stretch3_quest_teleop.config import TeleopConfig, default_config_path
from stretch3_quest_teleop.doctor import run_doctor


def test_core_doctor():
    result = run_doctor(TeleopConfig.load(default_config_path()))
    assert result["ok"]
    assert {item["name"] for item in result["checks"]} >= {
        "python",
        "config",
        "tls",
        "config_overlays",
        "base_speeds",
    }


def test_doctor_reports_the_speeds_actually_in_force():
    """The one number worth reading before the robot is allowed to move."""
    root = default_config_path().parent
    result = run_doctor(
        TeleopConfig.load(default_config_path(), [root / "overlay_base_locked.json"])
    )
    detail = {item["name"]: item["detail"] for item in result["checks"]}
    assert detail["config_overlays"] == "overlay_base_locked.json"
    assert detail["base_speeds"] == "NAV 0.0 m/s, 0.0 rad/s"
