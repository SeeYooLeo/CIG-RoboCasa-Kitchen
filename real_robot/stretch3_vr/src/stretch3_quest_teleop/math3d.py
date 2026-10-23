from __future__ import annotations

import math
from collections.abc import Sequence

from .models import Pose


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Matrix3 = tuple[tuple[float, float, float], ...]

# WebXR: +X right, +Y up, -Z forward.
# Stretch base: +X forward, +Y left, +Z up.
# This basis is for an operator standing behind the base, looking along its
# forward axis. See operator_basis() for standing anywhere else.
VR_TO_ROBOT: Matrix3 = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vec_add(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Sequence[float], scale: float) -> Vector3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def mat_vec(matrix: Matrix3, vector: Sequence[float]) -> Vector3:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def mat_mul(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def mat_transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def quat_normalize(q: Sequence[float]) -> Quaternion:
    norm = math.sqrt(sum(component * component for component in q))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / norm for component in q)  # type: ignore[return-value]


def quat_conjugate(q: Sequence[float]) -> Quaternion:
    x, y, z, w = quat_normalize(q)
    return (-x, -y, -z, w)


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return quat_normalize(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quat_to_matrix(q: Sequence[float]) -> Matrix3:
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def matrix_to_quat(matrix: Matrix3) -> Quaternion:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return quat_normalize(
            (
                (matrix[2][1] - matrix[1][2]) / s,
                (matrix[0][2] - matrix[2][0]) / s,
                (matrix[1][0] - matrix[0][1]) / s,
                0.25 * s,
            )
        )
    diagonal = [matrix[0][0], matrix[1][1], matrix[2][2]]
    index = diagonal.index(max(diagonal))
    if index == 0:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        q = (
            0.25 * s,
            (matrix[0][1] + matrix[1][0]) / s,
            (matrix[0][2] + matrix[2][0]) / s,
            (matrix[2][1] - matrix[1][2]) / s,
        )
    elif index == 1:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        q = (
            (matrix[0][1] + matrix[1][0]) / s,
            0.25 * s,
            (matrix[1][2] + matrix[2][1]) / s,
            (matrix[0][2] - matrix[2][0]) / s,
        )
    else:
        s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        q = (
            (matrix[0][2] + matrix[2][0]) / s,
            (matrix[1][2] + matrix[2][1]) / s,
            0.25 * s,
            (matrix[1][0] - matrix[0][1]) / s,
        )
    return quat_normalize(q)


def yaw_matrix(degrees: float) -> Matrix3:
    """Rotation about the robot's vertical axis, counter-clockwise from above."""
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    return (
        (cos, -sin, 0.0),
        (sin, cos, 0.0),
        (0.0, 0.0, 1.0),
    )


def operator_basis(operator_yaw_deg: float) -> Matrix3:
    """VR-to-robot basis for an operator who is not behind the base.

    `operator_yaw_deg` is the direction the operator faces, as a yaw in the
    robot's base frame: 0 looks along the base's forward axis, positive turns
    counter-clockwise seen from above (toward the base's left).

    Stretch's arm extends along the base's -Y, its right. An operator who
    stands alongside and faces the way the arm points is at -90, and for them
    "push my hand forward" must extend the arm, not drive the base forward.
    Without this the operator has to mentally rotate every command by a
    quarter turn, which is exactly the kind of load that produces a wrong
    move when something needs to stop.
    """
    if not operator_yaw_deg:
        return VR_TO_ROBOT
    return mat_mul(yaw_matrix(operator_yaw_deg), VR_TO_ROBOT)


def quat_scale_angle(q: Sequence[float], scale: float) -> Quaternion:
    """Scale a rotation about its own axis, leaving the axis alone."""
    x, y, z, w = quat_normalize(q)
    if scale == 1.0:
        return (x, y, z, w)
    if w < 0.0:  # take the short way round before scaling the angle
        x, y, z, w = -x, -y, -z, -w
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    half_angle = math.acos(clamp(w, -1.0, 1.0)) * scale
    factor = math.sin(half_angle) / sin_half
    return quat_normalize((x * factor, y * factor, z * factor, math.cos(half_angle)))


def map_vr_quaternion_to_robot(
    q: Sequence[float], basis: Matrix3 = VR_TO_ROBOT
) -> Quaternion:
    rotation = quat_to_matrix(q)
    mapped = mat_mul(mat_mul(basis, rotation), mat_transpose(basis))
    return matrix_to_quat(mapped)


def relative_quaternion(anchor: Sequence[float], current: Sequence[float]) -> Quaternion:
    """Return the world-frame rotation that moves anchor to current."""
    return quat_multiply(current, quat_conjugate(anchor))


def map_relative_controller_pose(
    vr_anchor: Pose,
    vr_current: Pose,
    robot_anchor: Pose,
    translation_scale: float,
    basis: Matrix3 = VR_TO_ROBOT,
    rotation_scale: float = 1.0,
) -> Pose:
    position_delta_vr = vec_sub(vr_current.position, vr_anchor.position)
    position_delta_robot = vec_scale(
        mat_vec(basis, position_delta_vr), translation_scale
    )
    orientation_delta_vr = quat_scale_angle(
        relative_quaternion(vr_anchor.orientation, vr_current.orientation),
        rotation_scale,
    )
    orientation_delta_robot = map_vr_quaternion_to_robot(orientation_delta_vr, basis)
    return Pose(
        vec_add(robot_anchor.position, position_delta_robot),
        quat_multiply(orientation_delta_robot, robot_anchor.orientation),
    )


def quat_rotate(q: Sequence[float], vector: Sequence[float]) -> Vector3:
    return mat_vec(quat_to_matrix(q), vector)


def hmd_yaw_pitch_relative(anchor: Pose, current: Pose) -> tuple[float, float]:
    relative = relative_quaternion(anchor.orientation, current.orientation)
    forward = quat_rotate(relative, (0.0, 0.0, -1.0))
    yaw = math.atan2(forward[0], -forward[2])
    pitch = math.asin(clamp(forward[1], -1.0, 1.0))
    return yaw, pitch


def shape_axis(value: float, deadzone: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(scaled, value)


def shortest_angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi
