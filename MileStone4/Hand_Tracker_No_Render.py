import mediapipe as mp
import cv2
import pyautogui
import math

def main():
    # Initialize MediaPipe Pose, Hands, and Drawing Modules
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # Initializing the Detectors
    hand_detector = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # OpenCV VideoCapture
    cap = cv2.VideoCapture(0)

    #Flag for input control
    mouse_button_down = False

    screen_width, screen_height = pyautogui.size()

    pyautogui.FAILSAFE = False

    left_hand_pinched = False

    while cap.isOpened():

        # Read a frame from the webcam
        return_value, image = cap.read()
        if not return_value:
            break

        # Flipping the webcam input
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detecting Hands and Poses
        hand_results = hand_detector.process(image_rgb)
        pose_results = pose_detector.process(image_rgb)

        # Setting the rendering style of the points
        joint_styling = mp_drawing.DrawingSpec(color=(246, 100, 20), thickness=2, circle_radius=4)
        line_styling = mp_drawing.DrawingSpec(color=(248, 200, 0), thickness=2, circle_radius=2)

        #run if hands are found in the frame
        if hand_results.multi_hand_landmarks:

            if hand_results.multi_handedness[0].classification[0].label == 'Right':
                #gets the position of the right hand landmarks, and detects wether or not it's pinching
                right_wrist_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
                right_index_tip_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                right_thumb_tip_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                right_pinch_distance = math.sqrt((right_index_tip_location.x - right_thumb_tip_location.x)**2 + (right_index_tip_location.y - right_thumb_tip_location.y)**2)

                if right_pinch_distance <= 0.065 and not mouse_button_down:
                    pyautogui.mouseDown()
                    mouse_button_down = True
                elif right_pinch_distance >= 0.065 and mouse_button_down:
                    pyautogui.mouseUp()
                    mouse_button_down = False

                #applying linear scaling to the tracking result
                normalized_wrist_loc_x = (right_wrist_location.x - 0.20) * 1/0.6
                normalized_wrist_loc_y = (right_wrist_location.y - 0.20) * 1/0.6

                #Translating the trackign result into screen coordinates
                cursor_position_x = int(normalized_wrist_loc_x * screen_width)
                cursor_position_y = int(normalized_wrist_loc_y * screen_height)
                
                #moving the mouse to the screen coordinates
                pyautogui.moveTo(cursor_position_x, cursor_position_y, duration=0.1)

            else:
                #gets the position of the left hand landmarks, and detects wether or not it's pinching
                left_index_tip_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                left_thumb_tip_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
                left_pinch_distance = math.sqrt((left_index_tip_location.x - left_thumb_tip_location.x)**2 + (left_index_tip_location.y - left_thumb_tip_location.y)**2)

                if left_pinch_distance <= 0.065 and not left_hand_pinched:
                    pyautogui.hotkey('winleft', 'ctrlleft', 'o')
                    left_hand_pinched = True
                elif left_pinch_distance <= 0.065 and left_hand_pinched:
                    left_hand_pinched = False
        
        #closing the program if the user hits 'Esc'
        if cv2.waitKey(1) == 27:
            break

    # Cleanup after the program is done
    cap.release()

if __name__ == "__main__":
    main()