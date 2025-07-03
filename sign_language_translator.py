import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import pyttsx3
import time
from gtts import gTTS
import playsound
import uuid
from threading import Thread
from googletrans import Translator

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Tamil dictionary (still used for fallback)
tamil_dict = {
    "HELLO": "வணக்கம்",
    "HOW": "எப்படி",
    "ARE": "இருக்கிறீர்கள்",
    "YOU": "நீங்கள்",
    "END": "முடிவு"
}

# TTS fallback
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

gesture_name = ""
last_gesture = ""
sentence = []
last_gesture_time = 0
cooldown_seconds =0.5

translator = Translator()

# TTS helper in separate thread
def speak_tamil(text):
    try:
        tts = gTTS(text, lang='ta')
        filename = f"temp_{uuid.uuid4()}.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("TTS error:", e)

# For adaptive color
text_color_light = (0, 0, 0)      # Black
text_color_dark = (255, 255, 255) # White

fps_time = time.time()

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
                        sentence_en = " ".join(sentence)

                        # Translate entire sentence once
                        try:
                            translation = translator.translate(
                                sentence_en,
                                src='en',
                                dest='ta'
                            )
                            sentence_ta = translation.text
                            print("Tamil Translation:", sentence_ta)

                            # TTS in new thread
                            Thread(target=speak_tamil, args=(sentence_ta,)).start()

                        except Exception as e:
                            print("Translation error:", e)
                            engine.say(sentence_en)
                            engine.runAndWait()

                        sentence = []

                    else:
                        sentence.append(gesture_name)

                    last_gesture = gesture_name
                    last_gesture_time = current_time
    else:
        gesture_name = ""
        last_gesture = ""

    # Prepare sentence strings
    sentence_en = " ".join(sentence)
    sentence_ta = " ".join(
        [tamil_dict.get(w, w) for w in sentence]
    )

    # Adaptive text color
    roi = frame[0:50, 0:300]
    avg_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    text_color = text_color_light if avg_brightness > 127 else text_color_dark

    # Draw everything directly in OpenCV
    cv2.putText(
        frame,
        f"Sentence (TA): {sentence_ta}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2
    )
    cv2.putText(
        frame,
        f"Sentence (EN): {sentence_en}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2
    )
    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2
    )

    # FPS for debugging
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2
    )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
