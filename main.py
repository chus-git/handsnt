import cv2
import mediapipe as mp
import pygame
from datetime import datetime
import time

# Initialize MediaPipe and pygame
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

pygame.init()
sound = pygame.mixer.Sound("sound.mp3")

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Initialize variables
last_detection_time = 0
detection_delay = 2

def log_hand_detection():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open("hand_detection_log.txt", "a") as file:
        file.write(f"[{timestamp}] - Hand detected\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image for hands
    result = hands.process(rgb_frame)

    current_time = time.time()
    if result.multi_hand_landmarks and (current_time - last_detection_time) > detection_delay:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand connections on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Play the sound
            pygame.mixer.Sound.play(sound)

            # Log hand detection
            log_hand_detection()

            # Update the last detection time
            last_detection_time = current_time

    # Display the frame with detections
    cv2.imshow('Hand Detection', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
