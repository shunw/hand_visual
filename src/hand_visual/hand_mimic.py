#!/usr/bin/env python3
"""change to the script for hand mimic with static photo."""

import numpy as np
import time
import logging
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd

from pyzlg_dexhand.dexhand_interface import (
    LeftDexHand,
    RightDexHand,
    ControlMode,
    ZCANWrapper,
    HandFeedback,
)
from pyzlg_dexhand.dexhand_logger import DexHandLogger
from pyzlg_dexhand import JointCommand
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from hand_para_convert import FingerConvert


@dataclass
class TestResult:
    """Store test metrics for a joint movement"""

    timestamp: float
    command_angle: float
    actual_angle: float
    steady_state_angle: float  # Average angle after settling
    steady_state_error: float  # Error after settling
    settling_time: float  # Time to reach steady state
    error_msg: Optional[str] = None
    command_time: Optional[float] = None


@dataclass
class JointConfig:
    """Configuration for a joint including limits"""

    min_angle: float = 0.0
    max_angle: float = 90.0  # Default for most joints

@dataclass
class JointConfigNew:
    """Configuration for a joint including limits"""

    min_angle: float = 0.0
    max_angle: float = 90.0  # Default for most joints
    tar_angle: float = 0.0 # Default target is 0.0


JOINT_CONFIGS = {
    "th_rot": JointConfigNew(max_angle=150.0, tar_angle=0.0),  # Thumb rotation has extended range
    "th_mcp": JointConfigNew(tar_angle=45.0),  # Default 0-90
    "th_dip": JointConfigNew(tar_angle=15.0),
    "ff_spr": JointConfigNew(max_angle=30.0, tar_angle=0.0),  # Finger spread is limited
    "ff_mcp": JointConfigNew(tar_angle=30.0),
    "ff_dip": JointConfigNew(tar_angle=30.0), # last 20
    "mf_mcp": JointConfigNew(tar_angle=38.0), # last ver 12.0
    "mf_dip": JointConfigNew(tar_angle=30.0), # last 15
    "rf_mcp": JointConfigNew(tar_angle=35.0),
    "rf_dip": JointConfigNew(tar_angle=30.0), # last 15
    "lf_mcp": JointConfigNew(tar_angle=30.0),
    "lf_dip": JointConfigNew(tar_angle=30.0), # last 25
}

class DexHandTester:
    """Test sequence runner for dexterous hands with steady-state analysis"""

    def __init__(
        self,
        hand_names: List[str],
        log_dir: str = "dexhand_logs",
        settling_time: float = 1.0,
        n_samples: int = 5,
    ):
        """Initialize test runner

        Args:
            hand_names: List of hands to test ('left', 'right')
            log_dir: Directory for test logs
            settling_time: Time to wait for settling (seconds)
            n_samples: Number of samples to collect for steady state
        """
        self.settling_time = settling_time
        self.n_samples = n_samples

        # Validate hand selection
        valid_hands = {"left", "right"}
        if not all(hand in valid_hands for hand in hand_names):
            raise ValueError(f"Invalid hand selection. Must be 'left' or 'right'")

        # Create ZCAN interface
        self.zcan = ZCANWrapper()
        if not self.zcan.open():
            raise RuntimeError("Failed to open ZCAN device")

        # Initialize hands
        self.hands = {}
        for name in hand_names:
            hand_class = LeftDexHand if name == "left" else RightDexHand
            self.hands[name] = hand_class(self.zcan)
            if not self.hands[name].init():
                raise RuntimeError(f"Failed to initialize {name} hand")
                
            # Check firmware version
            versions = self.hands[name].get_firmware_versions()
            if versions:
                # Print firmware versions for this hand
                for joint, version in versions.items():
                    if version is not None:
                        logger.info(f"{name} hand joint {joint} firmware version: {version}")
                
                # Get unique versions
                unique_versions = set(v for v in versions.values() if v is not None)
                if len(unique_versions) > 1:
                    logger.error(f"{name} hand has mismatched firmware versions: {unique_versions}")
                    raise RuntimeError(f"{name} hand has mismatched firmware versions")
                elif len(unique_versions) == 0:
                    logger.error(f"Could not read firmware versions for {name} hand")
                    raise RuntimeError(f"Could not read firmware versions for {name} hand")
                else:
                    logger.info(f"{name} hand firmware version: {list(unique_versions)[0]}")
            
            logger.info(f"Initialized {name} hand")

        # Create results directory
        self.results_dir = Path(log_dir) / f"test_{time.strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = DexHandLogger(str(self.results_dir))

        # Initialize test results storage
        self.results = {
            "individual": {joint: [] for joint in JOINT_CONFIGS},
            "simultaneous": [],
            "consecutive": {"commands": [], "feedback": []},
        }
  
    def _collect_steady_state(self, target_angles: Dict[str, float], settling_time: Optional[float] = None) -> List[HandFeedback]:
        """Collect multiple feedback samples after settling

        Args:
            target_angles: Dictionary of joint angles
            settling_time: Optional override for settling time

        Returns:
            List of feedback samples
        """
        # Wait for settling
        time.sleep(settling_time or self.settling_time)

        # Collect samples
        samples = []
        for _ in range(self.n_samples):
            for name, hand in self.hands.items():
                hand.move_joints(**target_angles)
                feedback = hand.get_feedback()
                self.logger.log_feedback(feedback, name)
                samples.append(feedback)

        return samples

    def _compute_steady_state_metrics(self, joint_name: str, target_angle: float, samples: List[HandFeedback]) -> Tuple[float, float]:
        """Compute steady-state metrics for a joint

        Args:
            joint_name: Name of joint
            target_angle: Commanded angle
            samples: List of feedback samples

        Returns:
            Tuple of (steady_state_angle, steady_state_error)
        """
        # Extract angles for the joint from samples
        angles = []
        for feedback in samples:
            if joint_name in feedback.joints:
                angles.append(feedback.joints[joint_name].angle)

        if not angles:
            return float("nan"), float("nan")

        # Compute metrics
        ss_angle = angles[-1]  # Last sample is assumed to be steady state
        ss_error = target_angle - ss_angle

        return ss_angle, ss_error

    def _test_reset(self):
        """Reset all joints and analyze steady-state"""
        start_time = time.time()

        # Send reset command
        for name, hand in self.hands.items():
            hand.reset_joints()
            feedback = hand.get_feedback()
            self.logger.log_command(
                "reset_joints",
                {joint: 0.0 for joint in JOINT_CONFIGS},
                ControlMode.IMPEDANCE_GRASP,
                name,
                feedback,
            )

        # Collect steady state data
        samples = self._collect_steady_state({})

        # Analyze each joint
        for joint in JOINT_CONFIGS:
            ss_angle, ss_error = self._compute_steady_state_metrics(joint, 0.0, samples)

            self.results["individual"][joint].append(
                TestResult(
                    timestamp=start_time,
                    command_angle=0.0,
                    actual_angle=ss_angle,
                    steady_state_angle=ss_angle,
                    steady_state_error=ss_error,
                    settling_time=self.settling_time,
                    error_msg=None,
                )
            )

    def _single_move(self, joint_commands:dict):
        # this is to make a single move from one place to another. this is both O.K for the single thumb or five fingers. 
        vel_value = 1000
        joint_commands_new = {joint_name:JointCommand(position=pos, velocity=vel_value)
                              for joint_name, pos in joint_commands.items()}
        # print (f'in the single move, the current position is {joint_commands_new}')
        try:
            
            for name, hand in self.hands.items():
                hand.move_joints(**joint_commands_new, control_mode = ControlMode.IMPEDANCE_GRASP)
                # hand.move_joints(th_rot=JointCommand(position=80.0, velocity=vel_value),
                #                  th_mcp=JointCommand(position=25.0, velocity=vel_value),
                #                  th_dip=JointCommand(position=10.0, velocity=vel_value),
                #                  control_mode = ControlMode.CASCADED_PID)
                feedback = hand.get_feedback()
                hand.clear_errors(clear_all=True, use_broadcast=False)
                # self.logger.log_command(
                #     "simultaneous",
                #     joint_commands_new,
                #     ControlMode.IMPEDANCE_GRASP,
                #     name,
                #     feedback,
                # )#ControlMode.IMPEDANCE_GRASP,

        except Exception as e:
            logger.error(f"Error in simultaneous movement test: {e}")
            raise
    def _single_analysis(self, joint_commands:dict, start_time):
        '''
        joint_commands: dict, which show the joint name and the target angle. so, if the fingers back to zero, joint_commands also will shows it
        '''
        # this is module the analysis steps
        samples = self._collect_steady_state(joint_commands)
        for joint_name in JOINT_CONFIGS:
            ss_angle, ss_error = self._compute_steady_state_metrics(
                joint_name, joint_commands[joint_name], samples
            )

            result = TestResult(
                timestamp=start_time,
                command_angle=joint_commands[joint_name],
                actual_angle=samples[0].joints[joint_name].angle,
                steady_state_angle=ss_angle,
                steady_state_error=ss_error,
                settling_time=self.settling_time,
            )
            self.results["simultaneous"].append(result)

            logger.info(
                f"{joint_name}: Target={joint_commands[joint_name]:.1f}°, "
                f"Steady-state={ss_angle:.1f}°, Error={ss_error:.1f}°"
            )
    def _finger_return_2_zero(self):
        """from grip paper to original place"""
        logger.info("Release paper")
        # target_angle = 30.0  # Use moderate angle for all joints

        try:
            
            start_time = time.time()
            
            # Second movement: Return to zero
            logger.info("Returning all joints to zero")
            start_time = time.time()
            zero_commands = {name: 0.0 for name in JOINT_CONFIGS}

            self._single_move(zero_commands)

            # Collect steady state data for return to zero
            logger.info(f"Waiting {self.settling_time}s for settling...")
            # samples = self._collect_steady_state(zero_commands)

            # Analyze return to zero for each joint
            self._single_analysis(zero_commands, start_time)
  
            # # Add simultaneous movement plots to report
            # self._plot_simultaneous_results()

        except Exception as e:
            logger.error(f"Error in simultaneous movement test: {e}")
            raise
        
    def close(self):
        """Clean up resources"""
        try:
            # Close each hand
            for hand in self.hands.values():
                try:
                    hand.close()
                except Exception as e:
                    logger.error(f"Error closing hand: {e}")

            # Close logger
            try:
                self.logger.close()
            except Exception as e:
                logger.error(f"Error closing logger: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

  
class MimicForLive():
    def __init__(self):
        self.args = DefaultPara()
        self.tester = None
    
    def _hand_init_(self):
        # Initialize and run tests
        self.tester = DexHandTester(
            self.args.hands, self.args.log_dir, self.args.settling_time, self.args.n_samples
        )
        
        # connect the hand and pass the joints data to hands
        self.tester._test_reset()
        time.sleep(.5)

    def _transfer_landmark(self, land_mark:list)->dict[str, JointConfigNew]:
        '''from the vedio, the output are just landmark, need to convert the landmark into dexhand readable dict
        land_mark: list, with 21 points, and each points are x, y, z'''
        f = FingerConvert(land_mark)
        return f.update_joint_configs()
    
    def action(self, joints:dict[str, JointConfigNew]):
        try:  
            self.tester._single_move({name: config.tar_angle
                    for name, config in joints.items()})
            time.sleep(.5)
      
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}", exc_info=self.args.debug)

        finally:
            
            logger.info("Test completed")
    def _hand_close(self):
        if self.tester is not None:
            try:
                self.tester.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}", exc_info=self.args.debug)
        
    def run(self, land_mark:list):
        joint_commands = self._transfer_landmark(land_mark)
        self.action(joint_commands)

class DefaultPara:
    def __init__(self):
        self.hands = ['right']
        self.log_dir = 'dexhand_logs'
        self.settling_time = 1.0
        self.n_samples = 5
        self.debug = None

def main_live(land_mark:list):
    # in the live system, just use all the default setting for the easy usage
    

    hand = MimicForLive()
    hand.run(land_mark)
    
    
if __name__ == "__main__":
    
    main_live()