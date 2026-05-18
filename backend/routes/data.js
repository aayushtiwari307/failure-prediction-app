const express = require('express');
const router = express.Router();
const axios = require('axios');
const Data = require('../models/Data');

// Upload sensor data only
router.post('/upload', async (req, res) => {
  try {
    const { sensorValues } = req.body;
    if (!sensorValues || !Array.isArray(sensorValues)) {
      return res.status(400).json({ error: 'sensorValues must be an array of numbers' });
    }

    const newEntry = new Data({
      sensorValues,
      failureProbability: 0,
      riskLevel: 'Pending',
      status: 'Uploaded'
    });

    await newEntry.save();
    res.status(201).json({ message: 'Data uploaded successfully', data: newEntry });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get Prediction + Save to DB
router.post('/predict', async (req, res) => {
  try {
    const { sensorValues } = req.body;
    if (!sensorValues || !Array.isArray(sensorValues)) {
      return res.status(400).json({ error: 'sensorValues must be an array' });
    }

    const mlResponse = await axios.post('http://localhost:5001/predict', { sensorValues });

    const prediction = mlResponse.data;

    const newEntry = new Data({
      sensorValues,
      failureProbability: prediction.failureProbability,
      riskLevel: prediction.riskLevel || 'High Risk',
      status: 'Predicted'
    });

    await newEntry.save();

    res.json({
      failureProbability: prediction.failureProbability,
      riskLevel: prediction.riskLevel || 'High Risk'
    });
  } catch (err) {
    console.error('Prediction error:', err.message);
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
});

// Get all data
router.get('/all', async (req, res) => {
  try {
    const data = await Data.find().sort({ timestamp: -1 });
    res.json({ data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Clear all data - THIS IS THE IMPORTANT ONE
router.delete('/all', async (req, res) => {
  try {
    const result = await Data.deleteMany({});
    console.log(`Deleted ${result.deletedCount} records`);
    res.json({ message: 'All data cleared successfully', deletedCount: result.deletedCount });
  } catch (err) {
    console.error('Clear error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;