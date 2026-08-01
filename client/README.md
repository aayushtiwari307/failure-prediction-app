# Proactive Machine Failure Prediction System

A full-stack **Predictive Maintenance** web application that predicts the likelihood of machine failure using sensor data. Built to demonstrate modern full-stack development with Machine Learning integration.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview)

## ✨ Features

- Real-time failure probability prediction using 4 industrial sensors:
  - Temperature (°C)
  - Vibration (mm/s)
  - Pressure (psi)
  - Usage (%)
- Trained **RandomForest ML Model** with **93.2% accuracy**
- Risk classification (Low / Medium / High) with smart maintenance suggestions
- Interactive trend chart using Recharts
- Data persistence with **MongoDB Atlas**
- Export prediction history to **CSV**
- Clear all data functionality

## 🛠️ Tech Stack

### Frontend
- React.js
- Material-UI (MUI)
- Recharts (for visualization)

### Backend
- Node.js + Express.js
- MongoDB Atlas + Mongoose

### Machine Learning
- Python + Flask (Microservice)
- RandomForest Classifier (Scikit-learn)
- Trained on synthetic industrial dataset (12,000+ records)

### Other Tools
- Axios for API calls
- dotenv for environment variables

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/tiwariaayush004-ui/failure-prediction-app.git
cd failure-prediction-app