const express = require('express');
const router = express.Router();
const axios = require('axios');
const Prediction = require('../models/Prediction');

// Milestone 3A canonical route. Express validates the request shape and
// orchestrates; it NEVER computes riskScore/riskLevel itself - Flask (the
// ML service) is the sole authority on inference and risk policy.

const ML_URL = process.env.ML_SERVICE_URL || 'http://localhost:5001';
const ML_TIMEOUT_MS = 5000;

const EXPECTED_FIELDS = [
  'air_temperature_k',
  'process_temperature_k',
  'rotational_speed_rpm',
  'torque_nm',
  'tool_wear_min',
  'type',
];
const ALLOWED_TYPES = ['L', 'M', 'H'];

function validateBody(body) {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return 'Request body must be a JSON object';
  }
  const missing = EXPECTED_FIELDS.filter((f) => !(f in body));
  if (missing.length > 0) return `Missing required field(s): ${missing.join(', ')}`;

  const unexpected = Object.keys(body).filter((f) => !EXPECTED_FIELDS.includes(f));
  if (unexpected.length > 0) return `Unexpected field(s): ${unexpected.join(', ')}`;

  for (const field of EXPECTED_FIELDS) {
    const value = body[field];
    if (field === 'type') {
      if (typeof value !== 'string' || !ALLOWED_TYPES.includes(value)) {
        return `'type' must be one of ${JSON.stringify(ALLOWED_TYPES)}`;
      }
      continue;
    }
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return `'${field}' must be a finite number`;
    }
  }
  return null;
}

async function callMlService(body) {
  return axios.post(`${ML_URL}/predict`, body, { timeout: ML_TIMEOUT_MS });
}

function mlErrorToResponse(err) {
  if (err.code === 'ECONNABORTED') {
    return { status: 504, code: 'ML_TIMEOUT', message: 'Prediction service took too long to respond' };
  }
  if (err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND') {
    return { status: 503, code: 'ML_SERVICE_UNAVAILABLE', message: 'Prediction service is currently unavailable' };
  }
  if (err.response) {
    const status = err.response.status >= 500 ? 503 : 502;
    return { status, code: 'ML_SERVICE_ERROR', message: 'Prediction service returned an error' };
  }
  return { status: 503, code: 'ML_SERVICE_UNAVAILABLE', message: 'Prediction service is currently unavailable' };
}

// POST /api/predictions
router.post('/', async (req, res) => {
  const validationError = validateBody(req.body);
  if (validationError) {
    return res.status(422).json({ error: { code: 'VALIDATION_FAILED', message: validationError } });
  }

  let mlResponse;
  try {
    mlResponse = await callMlService(req.body);
  } catch (err) {
    const mapped = mlErrorToResponse(err);
    return res.status(mapped.status).json({ error: { code: mapped.code, message: mapped.message } });
  }

  const prediction = mlResponse.data;
  const hasRequiredShape =
    prediction &&
    typeof prediction.riskScore === 'number' &&
    typeof prediction.riskLevel === 'string' &&
    typeof prediction.modelVersion === 'string';

  if (!hasRequiredShape) {
    return res.status(502).json({
      error: { code: 'MALFORMED_ML_RESPONSE', message: 'Prediction service returned an unexpected response shape' },
    });
  }

  // Persist. If this fails, the ML call already succeeded but the record
  // is lost - report distinctly from an ML failure so the client (and
  // logs) can tell the two apart.
  let saved;
  try {
    saved = await Prediction.create({
      input: {
        airTemperatureK: req.body.air_temperature_k,
        processTemperatureK: req.body.process_temperature_k,
        rotationalSpeedRpm: req.body.rotational_speed_rpm,
        torqueNm: req.body.torque_nm,
        toolWearMin: req.body.tool_wear_min,
        type: req.body.type,
      },
      result: {
        riskScore: prediction.riskScore,
        riskLevel: prediction.riskLevel,
      },
      modelVersion: prediction.modelVersion,
      predictedAt: prediction.predictedAt ? new Date(prediction.predictedAt) : new Date(),
    });
  } catch (dbErr) {
    return res.status(503).json({
      error: {
        code: 'PERSISTENCE_FAILED',
        message: 'Prediction was computed but could not be saved',
      },
    });
  }

  return res.status(201).json({
    id: saved._id,
    prediction: {
      riskScore: saved.result.riskScore,
      riskLevel: saved.result.riskLevel,
    },
    modelVersion: saved.modelVersion,
    predictedAt: saved.predictedAt,
  });
});

// GET /api/predictions?page=1&limit=20
router.get('/', async (req, res) => {
  let page = parseInt(req.query.page, 10);
  let limit = parseInt(req.query.limit, 10);
  if (!Number.isInteger(page) || page < 1) page = 1;
  if (!Number.isInteger(limit) || limit < 1 || limit > 100) limit = 20;

  try {
    const [records, totalCount] = await Promise.all([
      Prediction.find().sort({ createdAt: -1 }).skip((page - 1) * limit).limit(limit),
      Prediction.countDocuments(),
    ]);
    return res.status(200).json({
      records,
      page,
      limit,
      totalCount,
      totalPages: Math.ceil(totalCount / limit),
    });
  } catch (err) {
    return res.status(503).json({ error: { code: 'DATABASE_UNAVAILABLE', message: 'Could not retrieve prediction history' } });
  }
});

// GET /api/predictions/:id
router.get('/:id', async (req, res) => {
  try {
    const record = await Prediction.findById(req.params.id);
    if (!record) {
      return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Prediction not found' } });
    }
    return res.status(200).json(record);
  } catch (err) {
    return res.status(400).json({ error: { code: 'INVALID_ID', message: 'Invalid prediction id' } });
  }
});

// DELETE /api/predictions/:id
router.delete('/:id', async (req, res) => {
  try {
    const deleted = await Prediction.findByIdAndDelete(req.params.id);
    if (!deleted) {
      return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Prediction not found' } });
    }
    return res.status(200).json({ id: req.params.id, deleted: true });
  } catch (err) {
    return res.status(400).json({ error: { code: 'INVALID_ID', message: 'Invalid prediction id' } });
  }
});

// DELETE /api/predictions - bulk reset, protected by shared admin secret.
// Not full authentication: this is a single-user demo app, and a shared
// secret for the one destructive bulk operation is the proportionate
// control for this scope (documented decision, see README).
router.delete('/', async (req, res) => {
  const adminKey = process.env.ADMIN_KEY;
  if (!adminKey || req.headers['x-admin-key'] !== adminKey) {
    return res.status(401).json({ error: { code: 'UNAUTHORIZED', message: 'Missing or invalid admin key' } });
  }
  await Prediction.deleteMany({});
  return res.status(200).json({ deleted: true });
});

module.exports = router;
