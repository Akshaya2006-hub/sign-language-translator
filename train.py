# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load CSV
df = pd.read_csv("gesture_data.csv")

# Extract features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
