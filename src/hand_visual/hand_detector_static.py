#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import numpy as np

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_RGB2BGR)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)): # idx should be how many hands be detected
        hand_landmarks = hand_landmarks_list[idx] # shows the 21 hand mark at once 
        handedness = handedness_list[idx] # this is to check left or right hand, and the score of it
        
        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return (annotated_image, hand_landmarks_list[0])

from hand_para_convert import calculate_angle_supple, FingerConvert
def main():
    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    # IMPORTANT: Replace 'hand.png' with the actual path to your hand image.
    img = 'hand2.png'
    image = mp.Image.create_from_file(img)

    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    (annotated_image, hand_landmarks) = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow("Hand Landmark Detector", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # STEP 6: check the para convert
    f = FingerConvert(hand_landmarks)
    index_angle_mcp = f._get_dexhand_mcp('ff')
    index_angle_dip = f._get_dexhand_dip('ff')
    hand_spread = f._get_finger_spr()
    # 3. Print or display the result
    print(f"Index Finger Joint Angle: {index_angle_dip:.2f}")
    print (f"Hand spread angle is {hand_spread}")


if __name__ == '__main__':
    main()
