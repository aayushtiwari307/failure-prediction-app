import React, { useState, useEffect } from 'react';
import {
  Container, Typography, TextField, Button, Box, Card, CardContent,
  Alert, Chip, MenuItem
} from '@mui/material';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const EMPTY_FORM = {
  air_temperature_k: '',
  process_temperature_k: '',
  rotational_speed_rpm: '',
  torque_nm: '',
  tool_wear_min: '',
  type: 'M',
};

// Backend/ML service is the sole authority on risk classification.
// This component intentionally contains NO threshold values (no 0.03,
// no 0.7, no 0.4) - it only renders whatever riskLevel the API returns.
const RISK_DISPLAY = {
  NORMAL: { label: 'NORMAL', color: 'success' },
  ACTION_REQUIRED: { label: 'ACTION REQUIRED', color: 'error' },
};

function App() {
  const [form, setForm] = useState(EMPTY_FORM);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchHistory(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleInputChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const fetchHistory = async (targetPage) => {
    try {
      const res = await axios.get(`${API_BASE}/api/predictions`, {
        params: { page: targetPage, limit: 20 },
      });
      setHistory(res.data.records || []);
      setPage(res.data.page || 1);
      setTotalPages(res.data.totalPages || 1);
    } catch (err) {
      setError('Failed to load history');
    }
  };

  const submitPrediction = async () => {
    setLoading(true);
    setError('');
    try {
      const body = {
        air_temperature_k: Number(form.air_temperature_k),
        process_temperature_k: Number(form.process_temperature_k),
        rotational_speed_rpm: Number(form.rotational_speed_rpm),
        torque_nm: Number(form.torque_nm),
        tool_wear_min: Number(form.tool_wear_min),
        type: form.type,
      };
      const res = await axios.post(`${API_BASE}/api/predictions`, body);
      setPrediction(res.data);
      setForm(EMPTY_FORM);
      fetchHistory(1);
    } catch (err) {
      const apiMessage = err.response?.data?.error?.message;
      setError(apiMessage || 'Prediction failed');
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg" style={{ marginTop: 30, marginBottom: 50 }}>
      <Typography variant="h3" align="center" gutterBottom color="primary" fontWeight="bold">
        Predictive Maintenance Risk Assessment
      </Typography>
      <Typography align="center" sx={{ mb: 3, color: '#666' }}>
        Trained on the public AI4I 2020 predictive-maintenance benchmark dataset (synthetic, industrially-motivated).
        Not real industrial telemetry.
      </Typography>

      <Card elevation={3} sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>Machine Operating Snapshot</Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <TextField name="air_temperature_k" label="Air Temperature (K)" type="number" value={form.air_temperature_k} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="process_temperature_k" label="Process Temperature (K)" type="number" value={form.process_temperature_k} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="rotational_speed_rpm" label="Rotational Speed (RPM)" type="number" value={form.rotational_speed_rpm} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="torque_nm" label="Torque (Nm)" type="number" step="0.1" value={form.torque_nm} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="tool_wear_min" label="Tool Wear (min)" type="number" value={form.tool_wear_min} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }} />
            <TextField name="type" label="Machine Type" select value={form.type} onChange={handleInputChange} sx={{ flex: 1, minWidth: 180 }}>
              <MenuItem value="L">Low (L)</MenuItem>
              <MenuItem value="M">Medium (M)</MenuItem>
              <MenuItem value="H">High (H)</MenuItem>
            </TextField>
          </Box>

          <Box sx={{ mt: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Button variant="contained" color="secondary" onClick={submitPrediction} disabled={loading}>
              Get Risk Assessment
            </Button>
            <Button variant="outlined" onClick={() => fetchHistory(page)}>Refresh History</Button>
          </Box>
        </CardContent>
      </Card>

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      {loading && <Alert severity="info" sx={{ mb: 3 }}>Processing...</Alert>}

      {prediction && (
        <Card elevation={3} sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Latest Assessment</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h4" color="primary">
                Risk score: {prediction.prediction.riskScore.toFixed(4)}
              </Typography>
              <Chip
                label={RISK_DISPLAY[prediction.prediction.riskLevel]?.label || prediction.prediction.riskLevel}
                color={RISK_DISPLAY[prediction.prediction.riskLevel]?.color || 'default'}
                size="large"
              />
            </Box>
            <Typography variant="body2" sx={{ mt: 1, color: '#666' }}>
              Model {prediction.modelVersion} · {new Date(prediction.predictedAt).toLocaleString()}
            </Typography>
            <Typography variant="body2" sx={{ mt: 2, fontStyle: 'italic', color: '#666' }}>
              This is an uncalibrated model risk score, not a guaranteed real-world probability of failure.
            </Typography>
          </CardContent>
        </Card>
      )}

      {history.length > 0 && (
        <Card elevation={3} sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Prediction History</Typography>
            {history.map((entry) => (
              <Box key={entry._id} sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #eee' }}>
                <Typography variant="body2">
                  {new Date(entry.createdAt).toLocaleString()} · risk score {entry.result.riskScore.toFixed(4)}
                </Typography>
                <Chip
                  label={RISK_DISPLAY[entry.result.riskLevel]?.label || entry.result.riskLevel}
                  color={RISK_DISPLAY[entry.result.riskLevel]?.color || 'default'}
                  size="small"
                />
              </Box>
            ))}
            <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
              <Button size="small" disabled={page <= 1} onClick={() => fetchHistory(page - 1)}>Previous</Button>
              <Typography variant="body2">Page {page} of {totalPages}</Typography>
              <Button size="small" disabled={page >= totalPages} onClick={() => fetchHistory(page + 1)}>Next</Button>
            </Box>
          </CardContent>
        </Card>
      )}

      <Typography align="center" sx={{ mt: 4, color: '#666' }}>
        Predictive Maintenance Risk Assessment · MERN + HistGradientBoosting + MongoDB
      </Typography>
    </Container>
  );
}

export default App;
