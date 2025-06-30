import os

if os.path.exists("gesture_model.pkl"):
    os.remove("gesture_model.pkl")
    print("Deleted broken model file.")
