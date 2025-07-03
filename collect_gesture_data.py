import cv2
import mediapipe as mp
import numpy as np
import csv
from collections import defaultdict

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

gesture_label = None

# CSV file for storing gesture data (UTF-8 for Tamil compatibility)
file = open("gesture_data_tamil.csv", mode="a", newline='', encoding="utf-8")
writer = csv.writer(file)

sample_counts = defaultdict(int)
feedback_message = ""

print("➡ Press ENTER to enter a gesture label in Tamil or English.")
print("➡ Press 's' to save the current frame.")
print("➡ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.extend([x, y])

    # Display current label
    label_text = f"Current Label: {gesture_label}" if gesture_label else "No Label Selected"
    cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display sample count
    if gesture_label:
        count_text = f"Samples: {sample_counts[gesture_label]}"
        cv2.putText(frame, count_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display feedback message
    if feedback_message:
        cv2.putText(frame, feedback_message, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Tamil Sign Language Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER key
        gesture_label = input("Enter gesture label (Tamil or English): ").strip()
        feedback_message = f"Gesture label set to '{gesture_label}'"
    elif key == ord('s'):
        if gesture_label and landmarks and len(landmarks) == 42:
            writer.writerow(landmarks + [gesture_label])
            sample_counts[gesture_label] += 1
            feedback_message = f"Sample saved for '{gesture_label}'"
        else:
            feedback_message = "❌ No hand or label found."
    elif key == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
