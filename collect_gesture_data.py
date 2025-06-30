import cv2
import mediapipe as mp
import numpy as np
<<<<<<< HEAD
import csv
from collections import defaultdict

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

gesture_label = None

# Open CSV file for appending
file = open("gesture_data.csv", mode="a", newline='')
writer = csv.writer(file)

sample_counts = defaultdict(int)
feedback_message = ""

print("Press ENTER to change gesture label, or q to quit.")

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

    # Show current label
    label_text = f"Current Label: {gesture_label}" if gesture_label else "No Label Selected"
    cv2.putText(frame, label_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show sample counts
    if gesture_label:
        count_text = f"Samples Collected: {sample_counts[gesture_label]}"
        cv2.putText(frame, count_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if feedback_message:
        cv2.putText(frame, feedback_message, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('\r') or key == 13:  # Enter key
        gesture_label = input("Enter new gesture label: ").strip().upper()
        feedback_message = f"Gesture label set to {gesture_label}"
    elif key == ord('s'):
        if gesture_label and landmarks and len(landmarks) == 42:
            row = landmarks + [gesture_label]
            writer.writerow(row)
            sample_counts[gesture_label] += 1
            feedback_message = f"Sample saved for {gesture_label}"
        else:
            feedback_message = "No hand detected or no label set."
    elif key == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
=======
import pandas as pd
import os
from tkinter import *
from tkinter import ttk, messagebox

# Setup gesture data file
CSV_FILE = "gesture_data.csv"
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[f"{i}_{axis}" for i in range(21) for axis in ('x', 'y', 'z')] + ['label']).to_csv(CSV_FILE, index=False)

# Load existing labels
def load_labels():
    try:
        df = pd.read_csv(CSV_FILE)
        return sorted(df['label'].unique())
    except:
        return []

# Mediapipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Collect data for a given label
def collect_gesture(label):
    cap = cv2.VideoCapture(0)
    count = 0
    collected = []

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)
                collected.append(row)
                count += 1

        cv2.putText(frame, f"Capturing: {count}/100", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Gesture Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        df = pd.DataFrame(collected)
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        messagebox.showinfo("Done", f"{count} samples saved for gesture '{label}'.")

# Start data collection
def start_collection():
    selected = gesture_var.get().strip()
    new_label = new_gesture_entry.get().strip()

    if new_label:
        label = new_label
    elif selected:
        label = selected
    else:
        messagebox.showwarning("Missing", "Please select or enter a gesture name.")
        return

    collect_gesture(label)

# -------- Tkinter UI --------
root = Tk()
root.title("Sign Language Gesture Collector")

gesture_var = StringVar()
gesture_options = load_labels()

Label(root, text="Select Gesture:").pack(pady=5)
gesture_menu = ttk.Combobox(root, textvariable=gesture_var, values=gesture_options, state="readonly")
gesture_menu.pack(pady=5)

Label(root, text="Or Add New Gesture:").pack(pady=5)
new_gesture_entry = Entry(root)
new_gesture_entry.pack(pady=5)

Button(root, text="Start Recording", command=start_collection, bg="#4CAF50", fg="white").pack(pady=10)
Button(root, text="Quit", command=root.destroy, bg="gray", fg="white").pack(pady=5)

root.mainloop()
>>>>>>> e02ff71f92040d2e1eb28e0374e358ef6b29af03
