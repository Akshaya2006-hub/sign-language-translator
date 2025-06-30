import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load gesture data
data = pd.read_csv("gesture_data.csv")

if data.shape[0] < 5:
    print("Not enough samples to train the model. Collect more data first.")
    exit()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
