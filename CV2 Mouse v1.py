import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the mouse controller
mouse = Controller()

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # BGR 2 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Flip on horizontal
    image = cv2.flip(image, 1)
    # Set flag
    image.flags.writeable = False
    # Detections
    results = hands.process(image)
    # Set flag to true
    image.flags.writeable = True
    # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Rendering results
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            # Get the position of the tip of the index finger
            x_index = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y_index = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # Get the position of the tip of the thumb
            x_thumb = int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
            y_thumb = int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])

            # Find the distance between the index finger and thumb
            distance = np.sqrt((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2)

            # If the distance is less than a certain amount, perform a click
            if distance < 40:
                mouse.click(Button.left, 1)
            else:
                # Move the mouse
                mouse.position = (x_index, y_index)

            # Draw the hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

    # Show the image
    cv2.imshow('AI Virtual Mouse', image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
