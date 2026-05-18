from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model.pkl')
print("✅ Trained RandomForest Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sensor_values = data.get('sensorValues')
        
        if not sensor_values or len(sensor_values) != 4:
            return jsonify({'error': 'Provide exactly 4 sensor values'}), 400

        # Predict using trained model
        features = np.array([sensor_values]).reshape(1, -1)
        probability = model.predict_proba(features)[0][1]   # Probability of failure

        # Determine risk level
        if probability > 0.75:
            risk_level = "High Risk"
        elif probability > 0.45:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        response = {
            "failureProbability": float(probability),
            "riskLevel": risk_level,
            "status": "success"
        }

        print(f"Prediction → {probability:.1%} | {risk_level} | Sensors: {sensor_values}")
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ML Service running with trained model"})

if __name__ == '__main__':
    print("🚀 Trained ML Service started on port 5001")
    print("   Predictions are now data-driven and realistic!")
    app.run(host='0.0.0.0', port=5001, debug=False)