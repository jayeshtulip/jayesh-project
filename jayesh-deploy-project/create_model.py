import os
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("app/model", exist_ok=True)

# Save model
joblib.dump(model, "app/model/model.pkl")

print("✅ Model trained and saved to app/model/model.pkl")
