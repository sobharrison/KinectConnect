import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        #changing the frames color to coincide with the mediapipe library and flipping the image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        #Processing the image to detect the hands, and toggling flags as need be
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #changing the color back to display later
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #print(results) #this is just some optional debug info

        #this renders the "skeleton" of the hands. fits the colorscheme from our "paper prototype"
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(246, 100, 20), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(248, 200, 0), thickness=2, circle_radius=2))

        #finally displaying the image
        cv2.imshow('Hand Tracking', image)

        #This closes the program if the user hits the 'q' key on their keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

#cleanup after the program is done
cap.release()
cv2.destroyAllWindows()