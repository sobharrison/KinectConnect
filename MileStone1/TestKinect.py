import cv2

# Create a VideoCapture object for the Kinect camera
# Use the appropriate device index (usually 0 or 1) depending on setup

cap = cv2.VideoCapture(1)  # You might need to use 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("Error: Could not open Kinect camera.")
    exit()

while cap.isOpened():
    # Read a frame from the Kinect camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the RGB frame
    cv2.imshow('Kinect RGB Camera', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()