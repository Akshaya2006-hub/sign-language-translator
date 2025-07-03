import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load gesture data (UTF-8 for Tamil)
data = pd.read_csv("gesture_data_tamil.csv", encoding="utf-8")

if data.shape[0] < 5:
    print("Not enough samples to train the model. Collect more data first.")
    exit()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Optional: Show class distribution
print("Samples per gesture:\n", y.value_counts())

# Split data for testing (optional but useful)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Check training accuracy
accuracy = model.score(X_test, y_test)
print(f"Model test accuracy: {accuracy:.2f}")

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as gesture_model.pkl")
