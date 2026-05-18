import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Generating realistic training data for failure prediction...")

np.random.seed(42)
n = 12000

# Generate realistic sensor data
temp = np.random.normal(35, 12, n)
vibration = np.random.normal(1.8, 1.2, n)
pressure = np.random.normal(105, 35, n)
usage = np.random.normal(72, 22, n)

# Realistic failure logic
score = (
    0.42 * (temp > 48) +
    0.38 * (vibration > 3.5) +
    0.32 * ((pressure > 165) | (pressure < 55)) +
    0.28 * (usage > 92)
)

failure_prob = np.clip(score + np.random.uniform(-0.12, 0.22, n), 0.05, 0.96)
failure_label = (failure_prob > 0.5).astype(int)

df = pd.DataFrame({
    'temp': temp,
    'vibration': vibration,
    'pressure': pressure,
    'usage': usage,
    'failure': failure_label
})

# Train the model
X = df[['temp', 'vibration', 'pressure', 'usage']]
y = df['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"Accuracy: {accuracy:.1%} ({accuracy*100:.1f}%)")
print(f"Features: Temperature, Vibration, Pressure, Usage")

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl ✅")
print("You can now run app.py")