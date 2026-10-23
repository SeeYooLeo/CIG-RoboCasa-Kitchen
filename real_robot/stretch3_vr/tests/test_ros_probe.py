"""Check graph discovery and read-only probing with a simulated ROS graph."""
import json
import sys
import types

import pytest

from stretch3_quest_teleop.config import TeleopConfig, default_config_path, merge
from stretch3_quest_teleop.doctor import probe_ros2_graph


class FakeImage:
    def __init__(self, width, height, encoding="rgb8"):
        self.width = width
        self.height = height
        self.encoding = encoding


class FakeNode:
    def __init__(self, topics, services, actions, images):
        self._topics = topics
        self._services = services
        self._actions = actions
        self._images = images
        self.subscriptions = []
        self.live_subscriptions = 0
        self.destroyed = False

    def get_topic_names_and_types(self):
        return [(name, ["some/msg/Type"]) for name in self._topics]

    def get_service_names_and_types(self):
        return [(name, ["some/srv/Type"]) for name in self._services]

    def create_subscription(self, message_type, topic, callback, depth):
        subscription = types.SimpleNamespace(topic=topic, callback=callback)
        self.subscriptions.append(subscription)
        self.live_subscriptions += 1
        return subscription

    def destroy_subscription(self, subscription):
        self.live_subscriptions -= 1

    def destroy_node(self):
        self.destroyed = True

    def deliver(self):
        for subscription in self.subscriptions:
            image = self._images.get(subscription.topic)
            if image is not None:
                subscription.callback(image)

    # Anything that could command the robot must never be touched.
    def create_publisher(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("the probe must not create a publisher")

    def create_client(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("the probe must not create a service client")


@pytest.fixture
def fake_ros(monkeypatch):
    state = {"node": None, "init": 0, "shutdown": 0, "ok": False}

    def install(topics, services, actions, images):
        node = FakeNode(topics, services, actions, images)
        state["node"] = node

        rclpy = types.ModuleType("rclpy")
        rclpy.ok = lambda: state["ok"]

        def init():
            state["init"] += 1
            state["ok"] = True

        def shutdown():
            state["shutdown"] += 1
            state["ok"] = False

        rclpy.init = init
        rclpy.shutdown = shutdown
        rclpy.create_node = lambda name: node
        rclpy.spin_once = lambda target, timeout_sec=0.0: node.deliver()

        action_module = types.ModuleType("rclpy.action")
        action_module.get_action_names_and_types = lambda target: [
            (name, ["some/action/Type"]) for name in actions
        ]
        rclpy.action = action_module

        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msgs_msg.Image = FakeImage
        sensor_msgs.msg = sensor_msgs_msg

        for name, module in (
            ("rclpy", rclpy),
            ("rclpy.action", action_module),
            ("sensor_msgs", sensor_msgs),
            ("sensor_msgs.msg", sensor_msgs_msg),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return state

    return install


def _config(**robot_overrides):
    with default_config_path().open(encoding="utf-8") as stream:
        raw = json.load(stream)
    return TeleopConfig(merge(raw, {"robot": robot_overrides}) if robot_overrides else raw)


def _healthy_graph(config):
    robot = config.robot
    topics = [
        str(robot["cmd_vel_topic"]),
        str(robot["joint_states_topic"]),
        str(robot["odom_topic"]),
        str(robot["mode_topic"]),
    ]
    services = [
        str(robot["navigation_mode_service"]),
        str(robot["stop_service"]),
    ]
    actions = [str(robot["trajectory_action"])]
    return topics, services, actions


def _by_name(checks):
    return {item["name"]: item for item in checks}


def test_a_complete_graph_passes_and_reports_each_camera(fake_ros):
    config = _config()
    topics, services, actions = _healthy_graph(config)
    images = {
        str(config.robot["head_camera_topic"]): FakeImage(640, 480),
        str(config.robot["gripper_camera_topic"]): FakeImage(480, 640, "bgr8"),
    }
    state = fake_ros(topics, services, actions, images)

    checks = _by_name(probe_ros2_graph(config, 0.1))

    for name in (
        "ros2_topic:cmd_vel",
        "ros2_topic:joint_states",
        "ros2_topic:odom",
        "ros2_topic:mode",
        "ros2_service:navigation_mode",
        "ros2_service:stop_the_robot",
        "ros2_action:follow_joint_trajectory",
    ):
        assert checks[name]["ok"], checks[name]

    assert "640x480 rgb8 (landscape)" in checks["camera:head"]["detail"]
    assert "Hz" in checks["camera:head"]["detail"]
    assert "480x640 bgr8 (portrait)" in checks["camera:gripper"]["detail"]
    assert not checks["camera:navigation"]["ok"]
    assert checks["camera:navigation"]["required"] is False
    assert config.robot["navigation_camera_topic"] in checks["camera:navigation"]["detail"]
    assert checks["camera:panels"]["ok"]
    assert "left panel from head" in checks["camera:panels"]["detail"]

    node = state["node"]
    assert node.live_subscriptions == 0, "camera subscriptions must be torn down"
    assert node.destroyed
    assert state["init"] == 1 and state["shutdown"] == 1


def test_the_navigation_camera_covers_for_a_missing_head(fake_ros):
    config = _config()
    topics, services, actions = _healthy_graph(config)
    fake_ros(
        topics,
        services,
        actions,
        {
            str(config.robot["navigation_camera_topic"]): FakeImage(640, 480),
            str(config.robot["gripper_camera_topic"]): FakeImage(640, 480),
        },
    )
    checks = _by_name(probe_ros2_graph(config, 0.1))
    assert checks["camera:panels"]["ok"]
    assert "left panel from navigation" in checks["camera:panels"]["detail"]


def test_a_missing_gripper_camera_fails_the_required_pair_check(fake_ros):
    config = _config()
    topics, services, actions = _healthy_graph(config)
    fake_ros(
        topics,
        services,
        actions,
        {str(config.robot["head_camera_topic"]): FakeImage(640, 480)},
    )
    checks = _by_name(probe_ros2_graph(config, 0.1))
    assert checks["camera:panels"]["required"]
    assert not checks["camera:panels"]["ok"]
    assert "gripper" in checks["camera:panels"]["detail"]


def test_a_missing_action_server_is_named_not_just_flagged(fake_ros):
    config = _config()
    topics, services, _ = _healthy_graph(config)
    fake_ros(topics, services, [], {})
    checks = _by_name(probe_ros2_graph(config, 0.0))
    failed = checks["ros2_action:follow_joint_trajectory"]
    assert not failed["ok"] and failed["required"]
    assert str(config.robot["trajectory_action"]) in failed["detail"]


def test_a_configured_rotation_is_reported_next_to_the_picture(fake_ros):
    """So the printout says what arrives and what the bridge will do with it."""
    config = _config(camera_rotate_deg={"navigation": 0, "head": 90, "gripper": 0})
    topics, services, actions = _healthy_graph(config)
    fake_ros(
        topics,
        services,
        actions,
        {
            str(config.robot["head_camera_topic"]): FakeImage(640, 480),
            str(config.robot["gripper_camera_topic"]): FakeImage(640, 480),
        },
    )
    checks = _by_name(probe_ros2_graph(config, 0.1))
    assert "bridge rotates 90 deg clockwise" in checks["camera:head"]["detail"]
    assert "rotates" not in checks["camera:gripper"]["detail"]


def test_a_probe_of_zero_seconds_checks_the_graph_and_skips_the_cameras(fake_ros):
    config = _config()
    topics, services, actions = _healthy_graph(config)
    state = fake_ros(topics, services, actions, {})
    checks = _by_name(probe_ros2_graph(config, 0.0))
    assert checks["ros2_topic:cmd_vel"]["ok"]
    assert not any(name.startswith("camera:") for name in checks)
    assert state["node"].subscriptions == []
