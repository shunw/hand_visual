import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws the landmarks and the connections on the image."""
    annotated_image = np.copy(rgb_image)
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    # Loop through the detected hands
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            

    return annotated_image

def calculate_landmark_distance(landmarks1, landmarks2):
    """Calculates the Euclidean distance between two sets of landmarks."""
    if not landmarks1 or not landmarks2 or len(landmarks1) != len(landmarks2):
        return float('inf')

    total_distance = 0
    for lm1, lm2 in zip(landmarks1, landmarks2):
        total_distance += np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
    
    return total_distance / len(landmarks1)

from hand_mimic import main_live, MimicForLive

class VedioHandDetect:
    def __init__(self):
        # STEP 1: Create a HandLandmarker object.
        # The base options are used to configure the model.
        # The running mode is set to VIDEO, which means the model will process a stream of frames.
        # The result callback is a function that will be called when the model has processed a frame.
        # The model asset path is the path to the hand landmarker model file.
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                            running_mode=vision.RunningMode.VIDEO,
                                            num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

        self.timestamp = 0
        self.last_landmarks = None
        self.hand_stopped_start_time = None
        self.robot_action_taken = False
        self.MOVEMENT_THRESHOLD = 0.02 # This might need tuning

    def _get_hand_landmark(self, img):
        '''this is deal with the image from the live system '''
        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        # Convert the BGR image to RGB.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # STEP 4: Detect hand landmarks from the input image.
        self.timestamp += 1
        detection_result = self.detector.detect_for_video(mp_image, self.timestamp)
        return (mp_image, detection_result)

    def _hand_handle(self, detection_result):

        current_landmarks = None
        if detection_result.hand_landmarks:
            # For simplicity, we are using the first detected hand
            current_landmarks = detection_result.hand_landmarks[0]

        if current_landmarks and self.last_landmarks:
            distance = calculate_landmark_distance(current_landmarks, self.last_landmarks)
            
            if distance < self.MOVEMENT_THRESHOLD:
                if self.hand_stopped_start_time is None:
                    self.hand_stopped_start_time = time.time()
                
                if not self.robot_action_taken and (time.time() - self.hand_stopped_start_time) >= 3.0:
                    print("Hand stable for 3 seconds. Triggering robot action.")
                    self.robot_action_taken = True # Mark that action has been taken for this stable position
                    # Here you would transfer the hand landmark to the robot hand parameter.
                    return (self.robot_action_taken, current_landmarks)
                        

            else: # Hand moved
                self.hand_stopped_start_time = None
                self.robot_action_taken = False
        elif not current_landmarks:
            # No hand detected
            self.hand_stopped_start_time = None
            self.robot_action_taken = False

        return (self.robot_action_taken, current_landmarks)
    
    def _init_camera(self):
        # STEP 2: Open the camera.
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            return

        print("Camera opened successfully. Press 'q' to quit.")

        hand = MimicForLive()
        hand._hand_init_()
        try:
            while True:
                # STEP 3: Read a frame from the camera.
                success, img = self.cap.read()
                if not success:
                    print("Failed to grab frame.")
                    break

                # deal with the video image and detect the hand there
                mp_image, detection_result = self._get_hand_landmark(img)
                
                # judge the hand can be run or not, which is if the position is delayed with more than 3 seconds
                flag, self.last_landmarks = self._hand_handle(detection_result)
                if flag:
                    hand.run(self.last_landmarks)

                # STEP 5: Process the detection result. In this case, visualize it.
                if detection_result:
                    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
                    cv2.imshow("Hand Landmark Detector", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            hand._hand_close()
            self.cap.release()
            cv2.destroyAllWindows()
    




if __name__ == "__main__":
    v = VedioHandDetect()
    v._init_camera()
