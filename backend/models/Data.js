const mongoose = require('mongoose');

const dataSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    default: Date.now
  },
  sensorValues: {
    type: [Number],
    required: true
  },
  failureProbability: {
    type: Number,
    min: 0,
    max: 1,
    default: 0
  },
  failure: {
    type: Boolean,
    default: false
  },
  riskLevel: {
    type: String,
    default: ''
  },
  status: {
    type: String,
    default: ''
  }
});

module.exports = mongoose.models.Data || mongoose.model('Data', dataSchema);