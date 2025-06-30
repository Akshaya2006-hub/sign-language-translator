# live_test.py
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(int(lm.x * frame.shape[1]))
                landmarks.append(int(lm.y * frame.shape[0]))

            # Predict gesture
            prediction = model.predict([landmarks])[0]
            prob = model.predict_proba([landmarks]).max()

            # Show label
            cv2.putText(frame, f'{prediction} ({prob:.2f})', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
