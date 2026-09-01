// Deliberately NOT mocking Prediction here (unlike predictions.test.js) -
// these tests specifically verify that Mongoose's own schema validation
// (enum, min/max) is a real second line of defense against malformed ML
// output, independent of Express's own shape check. If Prediction were
// mocked, this guarantee couldn't be tested at all.
//
// NOTE: these tests still don't hit a real MongoDB - Prediction.create()
// with invalid data throws its Mongoose validation error before any
// network call is attempted, so no real database connection is required
// to exercise this specific guarantee. This is different from (and
// narrower than) real database integration testing.

const express = require('express');
const request = require('supertest');
const axios = require('axios');
const MockAdapter = require('axios-mock-adapter');
const predictionsRouter = require('./predictions');

const mock = new MockAdapter(axios);

function buildApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/predictions', predictionsRouter);
  return app;
}

const VALID_BODY = {
  air_temperature_k: 300, process_temperature_k: 310, rotational_speed_rpm: 1500,
  torque_nm: 40, tool_wear_min: 10, type: 'M',
};

afterEach(() => mock.reset());

test('out-of-contract riskScore (1.5 - a valid JS number but outside 0-1) is rejected by real Mongoose schema validation, not silently persisted', async () => {
  mock.onPost().reply(200, { modelVersion: 'v1', riskScore: 1.5, riskLevel: 'NORMAL', predictedAt: new Date().toISOString() });
  const res = await request(buildApp()).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(503);
  expect(res.body.error.code).toBe('PERSISTENCE_FAILED');
});

test('out-of-contract riskLevel string (a valid JS string but not NORMAL/ACTION_REQUIRED) is rejected by real Mongoose enum', async () => {
  mock.onPost().reply(200, { modelVersion: 'v1', riskScore: 0.5, riskLevel: 'SOMETHING_ELSE', predictedAt: new Date().toISOString() });
  const res = await request(buildApp()).post('/api/predictions').send(VALID_BODY);
  expect(res.status).toBe(503);
  expect(res.body.error.code).toBe('PERSISTENCE_FAILED');
});
