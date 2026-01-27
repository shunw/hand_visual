import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# STEP 1: Create a HandLandmarker object.
# The base options are used to configure the model.
# The running mode is set to VIDEO, which means the model will process a stream of frames.
# The result callback is a function that will be called when the model has processed a frame.
# The model asset path is the path to the hand landmarker model file.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.VIDEO,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws the landmarks and the connections on the image."""
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

def main():
    # STEP 2: Open the camera.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    timestamp = 0
    while True:
        # STEP 3: Read a frame from the camera.
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        # Convert the BGR image to RGB.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # STEP 4: Detect hand landmarks from the input image.
        timestamp += 1
        detection_result = detector.detect_for_video(mp_image, timestamp)

        # STEP 5: Process the detection result. In this case, visualize it.
        if detection_result:
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            cv2.imshow("Hand Landmark Detector", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
