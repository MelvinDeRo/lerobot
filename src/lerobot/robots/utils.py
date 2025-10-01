# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pprint import pformat
import math

from lerobot.robots import RobotConfig

from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so100_follower_end_effector":
        from .so100_follower import SO100FollowerEndEffector

        return SO100FollowerEndEffector(config)
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot

        return Stretch3Robot(config)
    elif config.type == "viperx":
        from .viperx import ViperX

        return ViperX(config)
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand

        return HopeJrHand(config)
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm

        return HopeJrArm(config)
    elif config.type == "bi_so100_follower":
        from .bi_so100_follower import BiSO100Follower

        return BiSO100Follower(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    else:
        raise ValueError(config.type)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    # if warnings_dict:
    #     logging.warning(
    #         "Relative goal position magnitude had to be clamped to be safe.\n"
    #         f"{pformat(warnings_dict, indent=4)}"
    #     )

    return safe_goal_positions

def ensure_authorized_goal_position(
    goal_pos: dict[str, float], max_position: float | dict[float]
) -> dict[str, float]:
    """Caps max distance between gripper and base"""

    if isinstance(max_position, float):
        max_position = dict.fromkeys(['x', 'y', 'z'], max_position)

    h0 = 44.45e-3
    a1 = 35e-3
    a2 = 112e-3
    a3 = 135e-3
    a4 = 42e-3
    c1 = 70e-3
    c2 = 27e-3
    c3 = 2e-3
    c5 = 125e-3
    
    # transform each goal position (which are in degrees) to radians
    goal_pos_rad = { key: (angle) * math.pi / 180 for key, angle in goal_pos.items()}
    
    # get the angles of the goal position
    thetas = [goal_pos_rad['shoulder_pan'], goal_pos_rad['shoulder_lift'], goal_pos_rad['elbow_flex'], goal_pos_rad['wrist_flex']]
    # compute the goal position on x,y and z from the angles
    x = a1 * math.cos(thetas[0]) + a2 * math.sin(thetas[1]) * math.cos(thetas[0]) + c2 * math.cos(thetas[1]) * math.cos(thetas[0]) \
        + a3 * math.cos(thetas[1] + thetas[2]) * math.cos(thetas[0]) + c3 * math.sin(thetas[1] + thetas[2]) * math.cos(thetas[0]) \
        + (a4 + c5) * math.cos(thetas[1] + thetas[2] + thetas[3]) * math.cos(thetas[0])

    y = a1 * math.sin(thetas[0]) + a2 * math.sin(thetas[1]) * math.sin(thetas[0]) + c2 * math.cos(thetas[1]) * math.sin(thetas[0]) \
        + a3 * math.cos(thetas[1] + thetas[2]) * math.sin(thetas[0]) + c3 * math.sin(thetas[1] + thetas[2]) * math.sin(thetas[0]) \
        + (a4 + c5) * math.cos(thetas[1] + thetas[2] + thetas[3]) * math.sin(thetas[0])

    z = h0 + c1 + a2 * math.cos(thetas[1]) - c2 * math.sin(thetas[1]) - a3 * math.sin(thetas[1] + thetas[2]) \
        + c3 * math.cos(thetas[1] + thetas[2]) - (a4 + c5) * math.sin(thetas[1] + thetas[2] + thetas[3])
    
    # print(f"\nPosition X : {round(x, 2)}, Position y: {round(y, 2)}, Position z : {round(z, 2)}")
    print(f"\nPosition intermédiaire : {h0 + c1 + a2 * math.cos(thetas[1]) - c2 * math.sin(thetas[1])}, autre position : {h0 + c1 + a2 * math.cos(thetas[1]) - c2 * math.sin(thetas[1]) - a3 * math.sin(thetas[1] + thetas[2])}")

    # if the goal position is greater than the max position set it to the goal position on x, y and/or z
    if (math.fabs(x) > max_position['x'] or math.fabs(y) > max_position['y'] or math.fabs(z) > max_position['z']):
        return False
    return True