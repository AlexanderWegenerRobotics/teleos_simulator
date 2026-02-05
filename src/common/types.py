from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

class ComponentStatus(Enum):
    OK = auto()
    TIMEOUT = auto()
    ERROR = auto()

@dataclass
class PoseCommand:
    """Cartesian pose command."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qw, qx, qy, qz]
    timestamp: float

@dataclass
class JointState:
    """Joint-level state."""
    positions: np.ndarray
    velocities: np.ndarray
    torques: np.ndarray
    timestamp: float

@dataclass
class CameraStatus:
    active_camera: str
    fps: float
    dropped_frames: int