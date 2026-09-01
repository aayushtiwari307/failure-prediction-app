const mongoose = require('mongoose');

// Milestone 3A: replaces the old Data schema (untyped sensorValues array,
// ambiguous `failure` boolean that was never actually set by any route).
//
// This schema stores exactly three things, and nothing is invented:
//   1. input   - the machine measurements as submitted (immutable record)
//   2. result  - what the model predicted (riskScore + riskLevel), verbatim
//                from the ML service - Express does not recompute this
//   3. modelVersion - which model produced the prediction, for traceability
//
// There is deliberately NO "actualFailure" or similar field: the
// application has no source of real-world observed outcomes, and adding
// one would misrepresent a placeholder as ground truth.

const predictionSchema = new mongoose.Schema({
  input: {
    airTemperatureK: { type: Number, required: true },
    processTemperatureK: { type: Number, required: true },
    rotationalSpeedRpm: { type: Number, required: true },
    torqueNm: { type: Number, required: true },
    toolWearMin: { type: Number, required: true },
    type: { type: String, required: true, enum: ['L', 'M', 'H'] },
  },
  result: {
    riskScore: { type: Number, required: true, min: 0, max: 1 },
    riskLevel: { type: String, required: true, enum: ['NORMAL', 'ACTION_REQUIRED'] },
  },
  modelVersion: { type: String, required: true },
  predictedAt: { type: Date, required: true },
  createdAt: { type: Date, default: Date.now },
});

// Supports the history endpoint's default sort (newest first).
predictionSchema.index({ createdAt: -1 });
// Supports "show me all predictions from model version X" (traceability queries).
predictionSchema.index({ modelVersion: 1, createdAt: -1 });

module.exports = mongoose.models.Prediction || mongoose.model('Prediction', predictionSchema);
