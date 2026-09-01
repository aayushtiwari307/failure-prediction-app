// NOTE ON TEST STRATEGY: this sandbox has no network path to a real
// MongoDB binary (mongodb-memory-server needs to download one, and the
// download host isn't reachable from here). These tests therefore mock
// the Prediction model's Mongoose methods rather than exercising a real
// database. This verifies Express's request handling, ML orchestration,
// and error-mapping logic correctly, but it is NOT a substitute for a
// real database integration test - that must be run against actual
// MongoDB (e.g. MongoDB Atlas) at deploy/CI time before this is trusted
// as a verified persistence path.

const express = require('express');
const request = require('supertest');
const axios = require('axios');
const MockAdapter = require('axios-mock-adapter');
const predictionsRouter = require('./predictions');
const Prediction = require('../models/Prediction');

jest.mock('../models/Prediction');

const mock = new MockAdapter(axios);

function buildApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/predictions', predictionsRouter);
  return app;
}

const VALID_BODY = {
  air_temperature_k: 300,
  process_temperature_k: 310,
  rotational_speed_rpm: 1500,
  torque_nm: 40,
  tool_wear_min: 10,
  type: 'M',
};

const VALID_ML_RESPONSE = {
  modelVersion: 'ai4i-hgb-v1-20260830',
  riskScore: 0.81,
  riskLevel: 'ACTION_REQUIRED',
  thresholdPolicy: { threshold: 0.03, costRatioAssumed: 5 },
  predictedAt: '2026-08-30T00:00:00.000Z',
};

afterEach(() => {
  mock.reset();
  jest.clearAllMocks();
  delete process.env.ADMIN_KEY;
});

test('valid prediction is persisted with input + result + modelVersion, no fake outcome field', async () => {
  mock.onPost('http://localhost:5001/predict').reply(200, VALID_ML_RESPONSE);
  const fakeSaved = {
    _id: 'abc123',
    input: { airTemperatureK: 300 },
    result: { riskScore: 0.81, riskLevel: 'ACTION_REQUIRED' },
    modelVersion: 'ai4i-hgb-v1-20260830',
    predictedAt: new Date('2026-08-30T00:00:00.000Z'),
  };
  Prediction.create.mockResolvedValue(fakeSaved);

  const app = buildApp();
  const res = await request(app).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(201);
  expect(res.body.prediction.riskLevel).toBe('ACTION_REQUIRED');
  expect(res.body.id).toBe('abc123');

  // Verify what Express actually asked Mongoose to store - proves no
  // fake actualFailure/failure field is fabricated.
  const savedArg = Prediction.create.mock.calls[0][0];
  expect(savedArg.input.airTemperatureK).toBe(300);
  expect(savedArg.result.riskScore).toBe(0.81);
  expect(savedArg.modelVersion).toBe('ai4i-hgb-v1-20260830');
  expect(savedArg.actualFailure).toBeUndefined();
  expect(savedArg.failure).toBeUndefined();
});

test('missing field returns 422, nothing persisted, ML never called', async () => {
  const app = buildApp();
  const body = { ...VALID_BODY };
  delete body.torque_nm;
  const res = await request(app).post('/api/predictions').send(body);
  expect(res.status).toBe(422);
  expect(mock.history.post.length).toBe(0);
  expect(Prediction.create).not.toHaveBeenCalled();
});

test('ML unavailable returns 503, nothing persisted', async () => {
  mock.onPost('http://localhost:5001/predict').networkError();
  const app = buildApp();
  const res = await request(app).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(503);
  expect(Prediction.create).not.toHaveBeenCalled();
});

test('ML timeout returns 504, nothing persisted', async () => {
  mock.onPost('http://localhost:5001/predict').timeout();
  const app = buildApp();
  const res = await request(app).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(504);
  expect(Prediction.create).not.toHaveBeenCalled();
});

test('malformed ML response returns 502, nothing persisted', async () => {
  mock.onPost('http://localhost:5001/predict').reply(200, { modelVersion: 'v1' });
  const app = buildApp();
  const res = await request(app).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(502);
  expect(Prediction.create).not.toHaveBeenCalled();
});

test('DB write failure after successful ML call returns distinct PERSISTENCE_FAILED error', async () => {
  mock.onPost('http://localhost:5001/predict').reply(200, VALID_ML_RESPONSE);
  Prediction.create.mockRejectedValue(new Error('connection lost'));
  const app = buildApp();
  const res = await request(app).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(503);
  expect(res.body.error.code).toBe('PERSISTENCE_FAILED');
});

test('history pagination request shape is correct', async () => {
  const fakeRecords = Array.from({ length: 20 }, (_, i) => ({ _id: `id${i}` }));
  Prediction.find.mockReturnValue({
    sort: () => ({ skip: () => ({ limit: () => Promise.resolve(fakeRecords) }) }),
  });
  Prediction.countDocuments.mockResolvedValue(25);

  const app = buildApp();
  const res = await request(app).get('/api/predictions?page=1&limit=20');
  expect(res.status).toBe(200);
  expect(res.body.records.length).toBe(20);
  expect(res.body.totalCount).toBe(25);
  expect(res.body.totalPages).toBe(2);
});

test('invalid page/limit query params fall back to sane defaults, not errors', async () => {
  Prediction.find.mockReturnValue({
    sort: () => ({ skip: () => ({ limit: () => Promise.resolve([]) }) }),
  });
  Prediction.countDocuments.mockResolvedValue(0);
  const app = buildApp();
  const res = await request(app).get('/api/predictions?page=-5&limit=99999');
  expect(res.status).toBe(200);
  expect(res.body.page).toBe(1);
  expect(res.body.limit).toBe(20); // clamped, not 99999
});

test('empty history returns empty records array, not an error', async () => {
  Prediction.find.mockReturnValue({
    sort: () => ({ skip: () => ({ limit: () => Promise.resolve([]) }) }),
  });
  Prediction.countDocuments.mockResolvedValue(0);
  const app = buildApp();
  const res = await request(app).get('/api/predictions');
  expect(res.status).toBe(200);
  expect(res.body.records).toEqual([]);
});

test('GET nonexistent id returns 404', async () => {
  Prediction.findById.mockResolvedValue(null);
  const app = buildApp();
  const res = await request(app).get('/api/predictions/507f1f77bcf86cd799439011');
  expect(res.status).toBe(404);
});

test('DELETE single prediction calls findByIdAndDelete with correct id', async () => {
  Prediction.findByIdAndDelete.mockResolvedValue({ _id: 'abc123' });
  const app = buildApp();
  const res = await request(app).delete('/api/predictions/abc123');
  expect(res.status).toBe(200);
  expect(Prediction.findByIdAndDelete).toHaveBeenCalledWith('abc123');
});

test('DELETE nonexistent single prediction returns 404', async () => {
  Prediction.findByIdAndDelete.mockResolvedValue(null);
  const app = buildApp();
  const res = await request(app).delete('/api/predictions/abc123');
  expect(res.status).toBe(404);
});

test('bulk DELETE without admin key returns 401, deleteMany never called', async () => {
  const app = buildApp();
  const res = await request(app).delete('/api/predictions');
  expect(res.status).toBe(401);
  expect(Prediction.deleteMany).not.toHaveBeenCalled();
});

test('bulk DELETE with wrong admin key returns 401', async () => {
  process.env.ADMIN_KEY = 'real-secret';
  const app = buildApp();
  const res = await request(app).delete('/api/predictions').set('x-admin-key', 'wrong-guess');
  expect(res.status).toBe(401);
  expect(Prediction.deleteMany).not.toHaveBeenCalled();
});

test('bulk DELETE with correct admin key succeeds', async () => {
  process.env.ADMIN_KEY = 'test-secret';
  Prediction.deleteMany.mockResolvedValue({ deletedCount: 5 });
  const app = buildApp();
  const res = await request(app).delete('/api/predictions').set('x-admin-key', 'test-secret');
  expect(res.status).toBe(200);
  expect(Prediction.deleteMany).toHaveBeenCalledWith({});
});
