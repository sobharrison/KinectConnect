import mediapipe as mp
import cv2
import pyautogui
import math

# Initialize MediaPipe Pose, Hands, and Drawing Modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Initializing the Detectors
hand_detector = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow('Hand Tracking')

#Flag for input control
mouse_button_down = False

screen_width, screen_height = pyautogui.size()

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
    pose_results = pose_detector.process(image_rgb)

    # Setting the rendering style of the points
    joint_styling = mp_drawing.DrawingSpec(color=(246, 100, 20), thickness=2, circle_radius=4)
    line_styling = mp_drawing.DrawingSpec(color=(248, 200, 0), thickness=2, circle_radius=2)

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

        #draws the landmarks
        for landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS, joint_styling, line_styling)
    
    #runs if a body is found in the frame
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, joint_styling,
                                  line_styling)

    # Finally displaying the image
    cv2.imshow('Hand Tracking', image)

    #closing the program if the user hits 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Cleanup after the program is done
cap.release()
cv2.destroyAllWindows()