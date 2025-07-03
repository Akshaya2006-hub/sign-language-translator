import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tkinter import *
from tkinter import ttk, messagebox
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setup gesture data file
CSV_FILE = "gesture_data.csv"
COLUMNS = [f"{i}_{axis}" for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=COLUMNS).to_csv(CSV_FILE, index=False)

# Global to hold recorded data until saved
collected_data = []

# Load existing labels
def load_labels():
    try:
        df = pd.read_csv(CSV_FILE)
        if 'label' in df.columns:
            return sorted(df['label'].dropna().unique())
        else:
            return []
    except:
        return []

# Mediapipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Collect data for a given label (doesn't save to CSV immediately)
def collect_gesture(label):
    global collected_data
    cap = cv2.VideoCapture(0)
    count = 0
    collected_data = []

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
                collected_data.append(row)
                count += 1

        cv2.putText(frame, f"Capturing: {count}/100", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Gesture Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected_data:
        messagebox.showinfo("Ready", f"{count} samples recorded for gesture '{label}'. Click 'Save Gesture' to store it.")
    else:
        messagebox.showwarning("No Data", "No gesture data was recorded.")

# Save recorded data to CSV and retrain model
def save_gesture():
    global collected_data
    if not collected_data:
        messagebox.showwarning("No Data", "No recorded data to save. Please record a gesture first.")
        return

    df = pd.DataFrame(collected_data, columns=COLUMNS)
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    collected_data.clear()

    update_dropdown()
    messagebox.showinfo("Saved", "Gesture data saved successfully.")

    # Optional: retrain the model immediately after saving
    retrain_model()

# Train and save the model using updated CSV
def retrain_model():
    try:
        df = pd.read_csv(CSV_FILE)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        joblib.dump(model, "gesture_model.pkl")
        accuracy = model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
    except Exception as e:
        messagebox.showerror("Training Error", f"Failed to train model:\n{str(e)}")

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

# Delete gesture from CSV
def delete_gesture(label_to_delete):
    if not os.path.exists(CSV_FILE):
        messagebox.showerror("Error", "No gesture data found.")
        return

    df = pd.read_csv(CSV_FILE)
    if label_to_delete not in df['label'].unique():
        messagebox.showinfo("Info", f"No gesture '{label_to_delete}' found.")
        return

    df = df[df['label'] != label_to_delete]
    df.to_csv(CSV_FILE, index=False)

    messagebox.showinfo("Deleted", f"Gesture '{label_to_delete}' has been deleted.")
    update_dropdown()

# Update dropdown list after deletion or addition
def update_dropdown():
    gesture_menu['values'] = load_labels()
    gesture_var.set('')

# -------- Tkinter UI --------
root = Tk()
root.title("Sign Language Gesture Collector")
root.geometry("400x370")

gesture_var = StringVar()
gesture_options = load_labels()

Label(root, text="Select Gesture:").pack(pady=5)
gesture_menu = ttk.Combobox(root, textvariable=gesture_var, values=gesture_options, state="readonly")
gesture_menu.pack(pady=5)

Label(root, text="Or Add New Gesture:").pack(pady=5)
new_gesture_entry = Entry(root)
new_gesture_entry.pack(pady=5)

Button(root, text="Start Recording", command=start_collection, bg="#4CAF50", fg="white").pack(pady=10)
Button(root, text="Save Gesture", command=save_gesture, bg="#2196F3", fg="white").pack(pady=5)
Button(root, text="Delete Gesture", command=lambda: delete_gesture(gesture_var.get()), bg="red", fg="white").pack(pady=5)
Button(root, text="Quit", command=root.destroy, bg="gray", fg="white").pack(pady=5)

root.mainloop()
