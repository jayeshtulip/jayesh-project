import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dummy data
X = np.random.rand(100, 3)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

print("Training model...")
model = LogisticRegression()
model.fit(X, y)

# Robust model path (no matter where the script is run from)
model_dir = os.path.join(os.path.dirname(__file__), "app", "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(model, model_path)

print(f"✅ Model saved to: {model_path}")
