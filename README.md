# Purpose: Capture Human Hand for Robot Hand Mimic

## step 1: to detect the hand from the camera or picture

## step 2: to convert the hand marker into the parameter of robot hand


# current status

- [x] complete the hand detect
- [ ] convert the hand marker data into the parameter of robot hand
    - the work logic will be three steps: 
    1. check how to convert with one figure (except the thumb)
    2. check how to do with the 4 fingers spread
    3. check how to covert the thumb
    4. check with all the figure angles with different shape, if it still can have a good mimic

### convert with the four fingers
- dip? to dexhand, this means the whole finger 斜度, this is all the finger mcp (5, 9, 13, 17) to wrist (0), vs the mcp to pip (6, 10, 14, 18) angles
- mcp? to dexhand, this means how curl finger is
    - the simple way would be the angle of mcp to pip vs pip to dip
    - or the way would be the average of two angles: the other angle is pip to dip vs dip to tip



# dexhand joint commands based as below
```
@dataclass
class JointConfig:
    """Configuration for a joint including limits"""

    min_angle: float = 0.0
    max_angle: float = 90.0  # Default for most joints


JOINT_CONFIGS = {
    "th_rot": JointConfig(max_angle=150.0),  # Thumb rotation has extended range
    "th_mcp": JointConfig(),  # Default 0-90 / # 拇指掌指关节弯曲（0-90度）
    "th_dip": JointConfig(),  # 拇指远端关节弯曲
    "ff_spr": JointConfig(max_angle=30.0),  # Finger spread is limited
    "ff_mcp": JointConfig(),
    "ff_dip": JointConfig(),
    "mf_mcp": JointConfig(),
    "mf_dip": JointConfig(),
    "rf_mcp": JointConfig(),
    "rf_dip": JointConfig(),
    "lf_mcp": JointConfig(),
    "lf_dip": JointConfig(),
}
```