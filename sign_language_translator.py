<<<<<<< HEAD
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import pyttsx3
import time  # Added for cooldown control

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize Mediapipe Hands
=======
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands module
>>>>>>> e02ff71f92040d2e1eb28e0374e358ef6b29af03
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
<<<<<<< HEAD
    max_num_hands=1,
=======
    max_num_hands=2,
>>>>>>> e02ff71f92040d2e1eb28e0374e358ef6b29af03
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

<<<<<<< HEAD
cap = cv2.VideoCapture(0)

gesture_name = ""
last_gesture = ""
sentence = []

# Cooldown logic setup
last_gesture_time = 0
cooldown_seconds = 1  # Minimum time between same gestures

=======
# Start video capture
cap = cv2.VideoCapture(0)

>>>>>>> e02ff71f92040d2e1eb28e0374e358ef6b29af03
while True:
    ret, frame = cap.read()
    if not ret:
        break

<<<<<<< HEAD
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.extend([x, y])

        if len(landmarks) == 42:
            landmarks_df = pd.DataFrame([landmarks])
            prediction = model.predict(landmarks_df)
            gesture_name = prediction[0]

            current_time = time.time()

            if gesture_name != last_gesture and gesture_name != "":
                if current_time - last_gesture_time > cooldown_seconds:
                    print("Predicted gesture:", gesture_name)

                    if gesture_name == "END":
                        sentence_text = " ".join(sentence)
                        if sentence_text.strip():
                            print("Speaking sentence:", sentence_text)
                            engine.say(sentence_text)
                            engine.runAndWait()
                            sentence = []
                    else:
                        sentence.append(gesture_name)

                    last_gesture = gesture_name
                    last_gesture_time = current_time
    else:
        gesture_name = ""
        last_gesture = ""

    # Display the sentence so far
    sentence_text = " ".join(sentence)
    cv2.putText(frame, f"Sentence: {sentence_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display current gesture
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

=======
    # Flip the frame horizontally for natural mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Example: extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.append((x, y))
            
            # Print landmarks to console
            print(landmarks)
            
            # Optional: calculate distance between thumb tip (4) and index tip (8)
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[8]
            distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            
            cv2.putText(frame, f"Distance: {int(distance)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
>>>>>>> e02ff71f92040d2e1eb28e0374e358ef6b29af03
    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
<<<<<<< HEAD
cv2.destroyAllWindows()
=======
cv2.destroyAllWindows()
>>>>>>> e02ff71f92040d2e1eb28e0374e358ef6b29af03
