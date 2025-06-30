import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# CSV file setup
CSV_FILE = 'gesture_data.csv'
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        writer.writerow(header)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

collect_data = False
current_label = ""

print("👋 Type the gesture label you want to collect and press Enter (e.g., Hello, Thank You)")
current_label = input("Enter label: ").strip()
print(f"[INFO] Press 's' to start/stop collecting data for label: {current_label}")
print("[INFO] Press 'n' to enter a new label, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
            if collect_data:
                flat = [coord for point in landmarks for coord in point]
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + flat)

    # Display current label on screen
    cv2.putText(frame, f"Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    status = "Recording" if collect_data else "Paused"
    cv2.putText(frame, f"Status: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        collect_data = not collect_data
        print(f"[INFO] {'Started' if collect_data else 'Stopped'} collecting for label: {current_label}")
    elif key == ord('n'):
        current_label = input("Enter new label: ").strip()
        print(f"[INFO] New label set: {current_label}")

cap.release()
cv2.destroyAllWindows()
