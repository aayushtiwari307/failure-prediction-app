import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import numpy as np

# Sample dataset for testing (in real app, load from CSV)
# Columns: temp, vibration, pressure, usage, failure (label)
sample_data = {
    'temp': [20, 25, 30, 35, 40, 22, 28, 32, 38, 45],
    'vibration': [0.5, 0.8, 1.2, 1.5, 2.0, 0.6, 0.9, 1.3, 1.6, 2.2],
    'pressure': [100, 105, 110, 115, 120, 102, 107, 112, 117, 122],
    'usage': [80, 85, 90, 95, 100, 82, 87, 92, 97, 102],
    'failure': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]  # 0 = no failure, 1 = failure
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Prepare data
X = df.drop('failure', axis=1)  # Features
y = df['failure']  # Labels

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=50,  # Small for quick training
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Training completed! Model accuracy on test set: {accuracy:.2f}')

# Save the model
joblib.dump(model, 'model.joblib')
print('Model saved as model.joblib')

# Test the saved model
loaded_model = joblib.load('model.joblib')
test_sample = np.array([[28, 1.0, 108, 88]])  # Sample input
test_pred = loaded_model.predict_proba(test_sample)[0][1]  # Probability of failure
print(f'Test prediction for [28, 1.0, 108, 88]: failure probability = {test_pred:.2f}')