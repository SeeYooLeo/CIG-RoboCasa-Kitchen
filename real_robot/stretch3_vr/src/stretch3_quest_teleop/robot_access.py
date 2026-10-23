"""Exclusive ownership for this package's hardware control entry points."""
import fcntl
import os
from pathlib import Path
import tempfile


class RobotControlLock:
    def __init__(self):
        path = Path(tempfile.gettempdir()) / f"stretch3-vr-control-{os.getuid()}.lock"
        self.stream = path.open("a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            raise RuntimeError("another VR or replay process owns robot control") from exc

    def close(self):
        if not self.stream.closed:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
