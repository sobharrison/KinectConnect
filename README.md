# KinectConnect
KinectConnect: Webcam Motion Sensing for Universal Access

*Work on in-between milestones wasn't uploaded to the repository separately; it was saved for the milestones that required a deliverable.*

## Milestone 1
This laid the groundwork for using `OpenCV`, a decision made irrespective of the hardware we ultimately chose.

It establishes getting webcam input through `OpenCV` and the initial loop that would later be used to collect tracking data.

## Milestone 3
This was when we decided to switch to using simple webcams over the Kinect, as the available libraries were all outdated and unusable.

Here, the use of the `MediaPipe` library was established to employ machine learning models for tracking user joints in their hands and upper body.

It simultaneously uses two models: one for the hands and the other for the arms, head, and torso.

## Milestone 4
Finally, this is when the tracking was refined and translated into user input that Windows could recognize through the `pyautogui` library.

The ability to differentiate between left and right hands was added, and the tracking data on the joints of each is differentiated.

Then, using the data of the left and right hands, it detects the position of the right hand and whether it is pinching or not, translating that into mouse movement and clicks.

The data of the left hand is used to determine whether or not it is pinching; if it is, it toggles an on-screen keyboard the user can utilize to type.
