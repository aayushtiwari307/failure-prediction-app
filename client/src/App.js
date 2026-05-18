import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, TextField, Button, Box, Card, CardContent, 
  LinearProgress, Alert, Chip 
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

function App() {
  const [sensorValues, setSensorValues] = useState({ temp: '', vibration: '', pressure: '', usage: '' });
  const [prediction, setPrediction] = useState(null);
  const [storedData, setStoredData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchStoredData();
  }, []);

  const handleInputChange = (e) => {
    setSensorValues({ ...sensorValues, [e.target.name]: e.target.value });
  };

  const getRiskColor = (prob) => prob > 0.7 ? 'error' : prob > 0.4 ? 'warning' : 'success';

  const getRiskSuggestion = (probability) => {
    if (probability > 0.7) return "⚠️ Immediate maintenance recommended. High chance of failure.";
    if (probability > 0.4) return "⚠️ Schedule maintenance within 7 days.";
    return "✅ Machine is running normally. Continue monitoring.";
  };

  const uploadData = async () => {
    setLoading(true);
    setError('');
    try {
      const values = Object.values(sensorValues).map(Number);
      await axios.post('http://localhost:5000/api/data/upload', { sensorValues: values });
      alert('✅ Data uploaded successfully!');
      fetchStoredData();
      setSensorValues({ temp: '', vibration: '', pressure: '', usage: '' });
    } catch (err) {
      setError('Upload failed');
    }
    setLoading(false);
  };

  const getPrediction = async () => {
    setLoading(true);
    setError('');
    try {
      const values = Object.values(sensorValues).map(Number);
      const response = await axios.post('http://localhost:5000/api/data/predict', { sensorValues: values });
      setPrediction(response.data);
      alert(`Failure Probability: ${(response.data.failureProbability * 100).toFixed(1)}% (${response.data.riskLevel})`);
      fetchStoredData();
      setSensorValues({ temp: '', vibration: '', pressure: '', usage: '' });
    } catch (err) {
      setError('Prediction failed');
    }
    setLoading(false);
  };

  const fetchStoredData = async () => {
    try {
      const res = await axios.get('http://localhost:5000/api/data/all');
      setStoredData(res.data.data || []);
    } catch (err) {
      setError('Failed to load history');
    }
  };

  const clearAllData = async () => {
    if (!window.confirm('⚠️ Delete ALL data from database permanently?')) return;
    setLoading(true);
    try {
      await axios.delete('http://localhost:5000/api/data/all');
      setStoredData([]);
      setPrediction(null);
      alert('✅ All data cleared successfully!');
    } catch (err) {
      alert('Clear failed. Try again.');
    }
    setLoading(false);
  };

  // New: Export to CSV
  const exportToCSV = () => {
    if (storedData.length === 0) {
      alert("No data to export!");
      return;
    }

    let csvContent = "Timestamp,Temperature,Vibration,Pressure,Usage,Failure Probability,Risk Level\n";
    
    storedData.forEach(entry => {
      const row = [
        entry.timestamp || '',
        entry.sensorValues?.[0] || '',
        entry.sensorValues?.[1] || '',
        entry.sensorValues?.[2] || '',
        entry.sensorValues?.[3] || '',
        entry.failureProbability ? (entry.failureProbability * 100).toFixed(1) + '%' : '',
        entry.riskLevel || ''
      ];
      csvContent += row.join(",") + "\n";
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `failure_predictions_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
    
    alert('✅ Data exported to CSV!');
  };

  const chartData = storedData.map(entry => ({
    date: entry.timestamp ? entry.timestamp.slice(5,10) : 'N/A',
    probability: entry.failureProbability ? parseFloat(entry.failureProbability.toFixed(2)) : 0,
    temp: entry.sensorValues?.[0] || 0,
  }));

  return (
    <Container maxWidth="lg" style={{ marginTop: 30, marginBottom: 50 }}>
      <Typography variant="h3" align="center" gutterBottom color="primary" fontWeight="bold">
        Proactive Failure Prediction System
      </Typography>

      {/* Input Section */}
      <Card elevation={3} sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>Enter Sensor Values</Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <TextField name="temp" label="Temperature (°C)" type="number" value={sensorValues.temp} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="vibration" label="Vibration (mm/s)" type="number" step="0.1" value={sensorValues.vibration} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="pressure" label="Pressure (psi)" type="number" value={sensorValues.pressure} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="usage" label="Usage (%)" type="number" value={sensorValues.usage} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
          </Box>

          <Box sx={{ mt: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button variant="contained" color="primary" onClick={uploadData} disabled={loading}>
              📤 Upload Data
            </Button>
            <Button variant="contained" color="secondary" onClick={getPrediction} disabled={loading}>
              🔮 Get Prediction & Save
            </Button>
            <Button variant="outlined" onClick={fetchStoredData}>🔄 Refresh</Button>
            <Button variant="outlined" color="success" onClick={exportToCSV}>
              📥 Export to CSV
            </Button>
            <Button variant="outlined" color="error" onClick={clearAllData}>
              🗑️ Clear All Data
            </Button>
          </Box>
        </CardContent>
      </Card>

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      {loading && <Alert severity="info" sx={{ mb: 3 }}>Processing...</Alert>}

      {/* Latest Prediction */}
      {prediction && (
        <Card elevation={3} sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Latest Prediction</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h3" color="primary">
                {(prediction.failureProbability * 100).toFixed(1)}%
              </Typography>
              <Chip label={prediction.riskLevel} color={getRiskColor(prediction.failureProbability)} size="large" />
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={prediction.failureProbability * 100} 
              sx={{ height: 12, borderRadius: 2, my: 2 }}
              color={getRiskColor(prediction.failureProbability)}
            />
            <Typography variant="body1" sx={{ fontStyle: 'italic', color: '#d32f2f' }}>
              {getRiskSuggestion(prediction.failureProbability)}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Chart */}
      {storedData.length > 0 && (
        <Card elevation={3} sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Failure Probability Trend</Typography>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="probability" stroke="#1976d2" strokeWidth={3} name="Failure Probability" yAxisId="left" />
                <Line type="monotone" dataKey="temp" stroke="#f57c00" strokeWidth={2} name="Temperature (°C)" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <Typography align="center" sx={{ mt: 4, color: '#666' }}>
        Proactive Failure Prediction System • MERN + XGBoost + MongoDB
      </Typography>
    </Container>
  );
}

export default App;