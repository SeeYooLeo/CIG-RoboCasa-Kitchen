from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RobotCommand, RobotState


class RobotBackend(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def apply(self, command: RobotCommand) -> dict:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def camera_diagnostics(self) -> dict:
        """Optional introspection for /healthz; backends need not implement it."""
        return {}

    def camera_frames(self) -> dict[str, bytes]:
        """One JPEG per panel slot. Missing slots are simply not drawn."""
        return {}

    def set_recording(self, enabled: bool) -> None:
        pass

    def recording_images(self) -> dict:
        return {}

    def apply_recorded_action(self, action: dict) -> None:
        raise NotImplementedError("this backend does not support recorded actions")
