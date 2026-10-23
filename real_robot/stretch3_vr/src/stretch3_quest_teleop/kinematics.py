from __future__ import annotations

from dataclasses import dataclass

from .models import Pose, RobotState


@dataclass(frozen=True)
class JointSolution:
    lift: float
    arm: float
    wrist_yaw: float
    wrist_pitch: float
    wrist_roll: float
    virtual_base_x: float


class StretchAIKinematics:
    """Optional adapter around stretch_ai's calibrated Pinocchio model."""

    def __init__(self) -> None:
        try:
            import numpy as np
            from stretch.motion.kinematics import (
                HelloStretchIdx,
                HelloStretchKinematics,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The ROS2 backend requires stretch_ai for FK/IK. Install the "
                "hello-robot/stretch_ai package in the robot environment."
            ) from exc
        self.np = np
        self.Idx = HelloStretchIdx
        self.model = HelloStretchKinematics(ik_type="pinocchio")

    def _configuration(
        self, state: RobotState, virtual_base_x: float
    ):
        q = self.np.zeros(self.model.dof, dtype=float)
        joints = state.joints
        q[self.Idx.BASE_X] = virtual_base_x
        q[self.Idx.LIFT] = float(joints.get("joint_lift", 0.5))
        q[self.Idx.ARM] = float(
            joints.get("wrist_extension", joints.get("joint_arm", 0.0))
        )
        q[self.Idx.WRIST_YAW] = float(joints.get("joint_wrist_yaw", 0.0))
        q[self.Idx.WRIST_PITCH] = float(joints.get("joint_wrist_pitch", 0.0))
        q[self.Idx.WRIST_ROLL] = float(joints.get("joint_wrist_roll", 0.0))
        q[self.Idx.HEAD_PAN] = state.head_pan
        q[self.Idx.HEAD_TILT] = state.head_tilt
        return q

    def forward(self, state: RobotState, virtual_base_x: float = 0.0) -> Pose:
        q = self._configuration(state, virtual_base_x)
        position, orientation = self.model.manip_fk(q)
        return Pose(
            tuple(float(x) for x in position),
            tuple(float(x) for x in orientation),
        )

    def solve(
        self,
        target: Pose,
        state: RobotState,
        virtual_base_x: float,
    ) -> JointSolution | None:
        q0 = self._configuration(state, virtual_base_x)
        # The manipulation solver expects its 9-DOF seed, not the full robot vector.
        q0_manip = self.model._to_manip_format(q0)
        q_manip, success, _ = self.model.manip_ik_solver.compute_ik(
            self.np.asarray(target.position, dtype=float),
            self.np.asarray(target.orientation, dtype=float),
            q0_manip,
            num_attempts=1,
        )
        if not success or q_manip is None:
            return None
        q = self.model._from_manip_format(q_manip, q0)
        return JointSolution(
            lift=float(q[self.Idx.LIFT]),
            arm=float(q[self.Idx.ARM]),
            wrist_yaw=float(q[self.Idx.WRIST_YAW]),
            wrist_pitch=float(q[self.Idx.WRIST_PITCH]),
            wrist_roll=float(q[self.Idx.WRIST_ROLL]),
            virtual_base_x=float(q[self.Idx.BASE_X]),
        )
