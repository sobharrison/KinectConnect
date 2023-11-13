import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose, Hands, and Drawing Modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

#Initializing the Deteectors
hand_detector = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)
pose_detector = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

#OpenCV VideoCapture
cap = cv2.VideoCapture(0)


while cap.isOpened():

    #Read a frame from the webcam
    return_value, image = cap.read()
    if not return_value:
        break

    #flipping the webcam input
    image = cv2.flip(image, 1)

    #Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #Detecting Hands and Poses
    hand_results = hand_detector.process(image_rgb)
    pose_results = pose_detector.process(image_rgb)

    #Rendering landmarks on the image
    joint_styling = mp_drawing.DrawingSpec(color = (246, 100, 20), thickness = 2, circle_radius = 4)
    line_styling = mp_drawing.DrawingSpec(color = (248, 200, 0), thickness = 2, circle_radius = 2)
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS, joint_styling, line_styling)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, joint_styling, line_styling)


    #finally displaying the image
    cv2.imshow('Hand Tracking', image)

    #This closes the program if the user hits the 'q' key on their keyboard
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#Cleanup after the program is done
cap.release()
cv2.destroyAllWindows()