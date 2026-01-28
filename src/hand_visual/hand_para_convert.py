import cv2
import mediapipe as mp
import numpy as np
from pydantic import BaseModel, model_validator
from enum import Enum
import csv



def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Re-using the angle function from before
def calculate_angle_supple(a, b, c): 
    """
    Calculates the angle at point B (in degrees) with the 180supplement.
    a, b, c are points as [x, y, z].

    How the Math Works
    To find the angle $theta$ at the middle joint (PIP), 
    we treat the finger segments as vectors vec{v_1} and vec{v_2}$. 
    The relationship is defined by:
    $$cos(theta) = frac{vec{v_1} cdot vec{v_2}}{|vec{v_1}| |vec{v_2}|}$$

    Where:
    $vec{v_1}$ is the vector from the middle joint to the knuckle.
    $vec{v_2}$ is the vector from the middle joint to the finger tip.
    $|vec{v}|$ represents the magnitude (length) of the vector.

    """
    a = np.array(a) # First point
    b = np.array(b) # Mid point (vertex)
    c = np.array(c) # End point

    # Create vectors
    ba = a - b
    cb = b - c 

    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, cb) / (np.linalg.norm(ba) * np.linalg.norm(cb))

    # Clip value to avoid errors due to floating point precision
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


# Re-using the angle function from before
def calculate_angle(a, b, c): 
    """
    Calculates the angle at point B (in degrees).
    a, b, c are points as [x, y, z].

    How the Math Works
    To find the angle $theta$ at the middle joint (PIP), 
    we treat the finger segments as vectors vec{v_1} and vec{v_2}$. 
    The relationship is defined by:
    $$cos(theta) = frac{vec{v_1} cdot vec{v_2}}{|vec{v_1}| |vec{v_2}|}$$

    Where:
    $vec{v_1}$ is the vector from the middle joint to the knuckle.
    $vec{v_2}$ is the vector from the middle joint to the finger tip.
    $|vec{v}|$ represents the magnitude (length) of the vector.

    """
    a = np.array(a) # First point
    b = np.array(b) # Mid point (vertex)
    c = np.array(c) # End point

    # Create vectors
    ba = a - b
    bc = c - b 

    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clip value to avoid errors due to floating point precision
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


class FingerName(str, Enum):
    '''enum for all the five fingers'''
    th = "th" # thumb
    ff = "ff" # index
    mf = "mf" # middle
    rf = "rf" # ring
    lf = "lf" # pinky

class FingerInd(BaseModel):
    '''to get the finger ind'''
    f_name:FingerName # this is the finger name
    mcp_ind:int
    pip_ind:int
    dip_ind:int
    tip_ind:int

def load_finger_ind(fl:str='lib/fing_ind.csv')->list[FingerInd]:
    '''this is to get the finger index for later usage'''
    with open(fl, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [FingerInd.model_validate(row) for row in reader]

class FingerConvert:
    def __init__(self, landmark_data:list):
        self.wrist_ind = 0
        self.finger_ind_ls = load_finger_ind()
        self.landmark = landmark_data

    def _get_dexhand_mcp(self, finger:str):
        '''this is for how curl the finger is
        - mcp? to dexhand, this means how curl finger is
            - the simple way would be the angle of mcp to pip vs pip to dip
            - or the way would be the average of two angles: the other angle is pip to dip vs dip to tip
        finger: int - which finger, choose from th/ ff/ mf/ rf/ lf
        '''
        cur_ind = [i for i in self.finger_ind_ls if i.f_name == finger][0]
        # We extract them as [x, y, z]
        pt3 = [self.landmark[cur_ind.dip_ind].x, self.landmark[cur_ind.dip_ind].y, self.landmark[cur_ind.dip_ind].z]
        pt2 = [self.landmark[cur_ind.pip_ind].x, self.landmark[cur_ind.pip_ind].y, self.landmark[cur_ind.pip_ind].z]
        pt1 = [self.landmark[cur_ind.mcp_ind].x, self.landmark[cur_ind.mcp_ind].y, self.landmark[cur_ind.mcp_ind].z]

        # 2. Calculate the angle
        return calculate_angle_supple(pt3, pt2, pt1)

    def _get_dexhand_dip(self, finger:str):
        '''this is for the angle between the finger vs palm
            dip? to dexhand, this means the whole finger 斜度, this is all the finger mcp (5, 9, 13, 17) to wrist (0), vs the mcp to pip (6, 10, 14, 18) angles
        finger: int - which finger, choose from th/ ff/ mf/ rf/ lf
        '''
        cur_ind = [i for i in self.finger_ind_ls if i.f_name == finger][0]
        # We extract them as [x, y, z]
        
        pt2 = [self.landmark[cur_ind.pip_ind].x, self.landmark[cur_ind.pip_ind].y, self.landmark[cur_ind.pip_ind].z]
        pt1 = [self.landmark[cur_ind.mcp_ind].x, self.landmark[cur_ind.mcp_ind].y, self.landmark[cur_ind.mcp_ind].z]
        pt0 = [self.landmark[self.wrist_ind].x, self.landmark[self.wrist_ind].y, self.landmark[self.wrist_ind].z]

        # 2. Calculate the angle
        return calculate_angle_supple(pt2, pt1, pt0)
    
    def _get_finger_spr(self):
        '''this is to get the spread of the palm. how large angles between fingers
        the logic is to calculate the angle between four fingerss
        ff_pip to ff_mcp vs ff_mcp to mf_mcp then this angle - 90
        rf_pip to rf_mcp vs rf_mcp to lf_mcp then this angle -90
        average these two angle to set the ff_spr

        note: this is only for the dexhand usage, due to the ff_spr is a common one and the degree is from 0 - 30
        '''
        ff_inf = [i for i in self.finger_ind_ls if i.f_name == 'ff'][0]
        mf_inf = [i for i in self.finger_ind_ls if i.f_name == 'mf'][0]
        rf_inf = [i for i in self.finger_ind_ls if i.f_name == 'rf'][0]
        lf_inf = [i for i in self.finger_ind_ls if i.f_name == 'lf'][0]

        # this is to get the three points for the angle1
        v1_p1 = [self.landmark[ff_inf.pip_ind].x, self.landmark[ff_inf.pip_ind].y, self.landmark[ff_inf.pip_ind].z]
        v1_p2 = [self.landmark[ff_inf.mcp_ind].x, self.landmark[ff_inf.mcp_ind].y, self.landmark[ff_inf.mcp_ind].z]
        v1_p3 = [self.landmark[mf_inf.mcp_ind].x, self.landmark[mf_inf.mcp_ind].y, self.landmark[mf_inf.mcp_ind].z]
        v1_angle = max(0, calculate_angle(v1_p1, v1_p2, v1_p3) - 90)

        # this is to get the three points for the angle1
        v2_p1 = [self.landmark[rf_inf.pip_ind].x, self.landmark[rf_inf.pip_ind].y, self.landmark[rf_inf.pip_ind].z]
        v2_p2 = [self.landmark[rf_inf.mcp_ind].x, self.landmark[rf_inf.mcp_ind].y, self.landmark[rf_inf.mcp_ind].z]
        v2_p3 = [self.landmark[lf_inf.mcp_ind].x, self.landmark[lf_inf.mcp_ind].y, self.landmark[lf_inf.mcp_ind].z]
        v2_angle = max(0, calculate_angle(v2_p1, v2_p2, v2_p3) - 90)
        # print (f'the v1 angle value is {v1_angle}, the v2 angle value is {v2_angle}')

        return min(30.0, (v1_angle + v2_angle)/2.0)

    def _get_th_rot(self):
        '''
        this is to calculate the angle of the thumb rotate
        the logic is: to calculate the th_mcp to ff_mcp vs ff_mcp to mf_mcp
        '''
        pass
        


if __name__ == '__main__':
    a = FingerConvert()
    a._get_dexhand_mcp('ff')    


    # if results.multi_hand_landmarks:
    # for hand_landmarks in results.multi_hand_landmarks:
    #     # 1. Get specific landmarks (index finger joints)
    #     # We extract them as [x, y, z]
    #     pt8 = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z]
    #     pt7 = [hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y, hand_landmarks.landmark[7].z]
    #     pt6 = [hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z]

    #     # 2. Calculate the angle
    #     index_angle = calculate_angle(pt8, pt7, pt6)

    #     # 3. Print or display the result
    #     print(f"Index Finger Joint Angle: {index_angle:.2f}")
    
