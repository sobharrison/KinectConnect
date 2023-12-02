import mediapipe as mp
import cv2
import pyautogui
import math

# Initialize MediaPipe Hands and Drawing Modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initializing the Detector
hand_detector = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# OpenCV VideoCapture
cap = cv2.VideoCapture(0)

#Flag for input control
mouse_button_down = False

#getting screen size
screen_width, screen_height = pyautogui.size()

#setting so the program doesn't auto-terminate
pyautogui.FAILSAFE = False

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

    #run if hands are found in the frame
    if hand_results.multi_hand_landmarks:

        #gets the position of the wrist
        wrist_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
        index_tip_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip_location = hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]

        pinch_distance = math.sqrt((index_tip_location.x - thumb_tip_location.x)**2 + (index_tip_location.y - thumb_tip_location.y)**2)

        if pinch_distance <= 0.065 and not mouse_button_down:
            pyautogui.mouseDown()
            mouse_button_down = True
        elif pinch_distance >= 0.065 and mouse_button_down:
            pyautogui.mouseUp()
            mouse_button_down = False

        #applying linear scaling to the tracking result
        normalized_wrist_loc_x = (wrist_location.x - 0.20) * 1/0.6
        normalized_wrist_loc_y = (wrist_location.y - 0.20) * 1/0.6

        #Translating the trackign result into screen coordinates
        cursor_position_x = int(normalized_wrist_loc_x * screen_width)
        cursor_position_y = int(normalized_wrist_loc_y * screen_height)

        #moving the mouse to the screen coordinates
        pyautogui.moveTo(cursor_position_x, cursor_position_y, duration=0.1)

# Cleanup after the program is done
cap.release()