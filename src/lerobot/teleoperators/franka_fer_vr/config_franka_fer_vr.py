"""
Configuration for Franka FER VR Teleoperator.

This module defines the configuration class for the Franka FER VR teleoperator,
including TCP connection settings, VR processing parameters, and IK solver options.
"""

from dataclasses import dataclass
from typing import List, Optional

from lerobot.teleoperators.config import TeleoperatorConfig


# Franka FR3 Joint Limits (in radians)
# Source: Franka FR3 Datasheet
# A1: -166° ~ 166°  → -2.897 ~ 2.897
# A2: -105° ~ 105°  → -1.833 ~ 1.833
# A3: -166° ~ 166°  → -2.897 ~ 2.897
# A4: -176° ~ -7°   → -3.072 ~ -0.122 (ASYMMETRIC!)
# A5: -165° ~ 165°  → -2.880 ~ 2.880
# A6: 25° ~ 265°    → 0.436 ~ 4.625 (ASYMMETRIC!)
# A7: -175° ~ 175°  → -3.054 ~ 3.054
FRANKA_FR3_JOINT_LIMITS = [
    -2.897,   # A1 min
    2.897,    # A1 max
    -1.833,   # A2 min
    1.833,    # A2 max
    -2.897,   # A3 min
    2.897,    # A3 max
    -3.072,   # A4 min (ASYMMETRIC!)
    -0.122,   # A4 max (ASYMMETRIC!)
    -2.880,   # A5 min
    2.880,    # A5 max
    0.436,    # A6 min (ASYMMETRIC!)
    4.625,    # A6 max (ASYMMETRIC!)
    -3.054,   # A7 min
    3.054,    # A7 max
]

# Franka FR3 Home Position (radians)
# Modified from default to be within safe range
FRANKA_FR3_HOME_POSITION = [
    -0.411,
    0.45,
    -0.072,
    0.725,
    0.094,
    0.048,
    0.258,
]

# [
#      0.0,    # J0: 0° - 居中
#     -0.3,   # J1: ~-17° - 轻微抬起
#     0.0,    # J2: 0° 
#     -1.5,   # J3: ~-86° - 前臂前伸
#     0.0,    # J4: 0°
#     0.8,    # J5: ~+46° - 手腕抬起
#     0.3     # J6: ~+17° - 末端轻微旋转
# ]

# [0, -0.785, 0, -2.356, 0, 1.571, 0.785]


@TeleoperatorConfig.register_subclass("franka_fer_vr")
@dataclass
class FrankaFERVRTeleoperatorConfig(TeleoperatorConfig):
    """
    Configuration for Franka FER VR teleoperator using C++ IK solver.
    
    For Franka FR3, use robot_type="fr3" to apply correct joint limits.
    """
    
    # Robot type selection
    robot_type: str = "fr3"  # "fr2" for Panda, "fr3" for Franka Research 3
    
    # TCP connection settings
    tcp_port: int = 8000
    setup_adb: bool = True  # Automatically setup adb reverse port forwarding
    
    # VR processing settings - SMOOTHING IS KEY FOR SAFE OPERATION!
    smoothing_factor: float = 0.99  # Higher = more smoothing (0-1). 0.92 = smooth, 0.7 = fast
    position_deadzone: float = 0.001  # 1mm deadzone to prevent drift
    orientation_deadzone: float = 0.03  # ~1.7 degrees deadzone
    
    # Workspace limits - restrict VR movement range
    max_position_offset: float = 0.50  # 50cm max workspace (safer for FR3)
    
    # IK solver settings
    manipulability_weight: float = 1.0
    neutral_distance_weight: float = 2.0
    current_distance_weight: float = 2.0
    joint_weights: Optional[List[float]] = None  # Will default to [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Q7 limits (can be configured for different end effectors)
    # For FR3 with default gripper: use full range
    # For FR3 with BiDexHand: use [-0.2, 1.9]
    q7_min: float = -3.0159  # Full Franka range (will be overridden by robot_type)
    q7_max: float = 3.0159
    
    # Debug settings
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize default values and apply FR3/FR2 limits based on robot_type."""
        # Set default joint weights if not provided
        if self.joint_weights is None:
            # Higher weights for base joints for stability
            self.joint_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Apply robot-specific Q7 limits
        if self.robot_type == "fr3":
            # FR3 full range
            self.q7_min = -3.0159
            self.q7_max = 3.0159
        elif self.robot_type == "fr2":
            # FR2 (Panda) full range
            self.q7_min = -2.8973
            self.q7_max = 2.8973
        elif self.robot_type == "bidexhand":
            # BiDexHand limited range
            self.q7_min = -0.2
            self.q7_max = 1.9