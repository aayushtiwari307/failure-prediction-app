# Proactive Machine Failure Prediction System

A full-stack **Predictive Maintenance** web application that predicts machine failure probability using industrial sensor data.

![Project Preview](https://via.placeholder.com/800x400/1976d2/ffffff?text=Proactive+Failure+Prediction+Dashboard)

## ✨ Features

- Predicts failure probability using 4 key sensors: **Temperature, Vibration, Pressure, Usage**
- Trained **RandomForest ML Model** with **93.2% accuracy**
- Real-time risk classification (Low / Medium / High Risk)
- Smart maintenance suggestions
- Interactive trend visualization
- Data stored permanently in **MongoDB Atlas**
- Export history to **CSV** file
- Clear all data functionality

## 🛠️ Tech Stack

- **Frontend**: React.js + Material-UI + Recharts
- **Backend**: Node.js + Express.js
- **Database**: MongoDB Atlas + Mongoose
- **Machine Learning**: Python + Flask + RandomForest (Scikit-learn)
- **Architecture**: MERN + Separate ML Microservice

## 🚀 How to Run Locally

### Prerequisites
- Node.js installed
- Python 3 + pip
- Git

### 1. Clone the repository
```bash
git clone https://github.com/tiwariaayush004-ui/failure-prediction-app.git
cd failure-prediction-app
