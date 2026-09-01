# Predictive Maintenance Risk Assessment System

A full-stack application that estimates machine failure risk from operating
sensor readings, trained and evaluated on a public benchmark dataset. Built
as a portfolio project demonstrating an end-to-end ML + web engineering
pipeline: data leakage analysis, model comparison, calibration evaluation,
cost-sensitive threshold selection, a validated model-serving API, and
persistence — with honest, documented limitations rather than inflated claims.

## What this is, and what it is not

This system estimates a **risk score** from the AI4I 2020 predictive
maintenance **benchmark dataset** — a public dataset UCI explicitly
describes as synthetic/simulated, designed to reflect industrial
predictive-maintenance data. **It is not trained on real production
telemetry**, and the results here should be read as a demonstration of
methodology, not a claim of real-world industrial performance.

The model output is called a **risk score**, not a "probability of
failure," because calibration was tested and did not hold up (see
Calibration below). The system also does not claim a future time
horizon ("fails within N days") — the dataset provides point-in-time
failure labels, not a time-to-failure target, so inventing a horizon
would be a false claim the data doesn't support.

## Architecture

```
React (client)
  ↓ POST/GET /api/predictions
Express (backend) — validates request shape, orchestrates, persists
  ↓ POST /predict (named JSON fields)
Flask (backend/ml-service) — sole authority on inference AND risk policy
  ↓
HistGradientBoostingClassifier (scikit-learn Pipeline)
  ↓
risk score + NORMAL/ACTION_REQUIRED
  ↓
MongoDB — stores input + result + model version per prediction
```

Express never recomputes risk thresholds. Flask is the single source of
truth for both the model's score and the NORMAL/ACTION_REQUIRED
classification; Express and React only forward/render it.

## Dataset and feature engineering

**Source**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601), UCI Machine Learning Repository. 10,000 rows, 3.39% positive (failure) rate.

**Features used** (5 numeric + 1 categorical):

| Feature | Column | Unit |
|---|---|---|
| `air_temperature_k` | Air temperature [K] | Kelvin |
| `process_temperature_k` | Process temperature [K] | Kelvin |
| `rotational_speed_rpm` | Rotational speed [rpm] | rpm |
| `torque_nm` | Torque [Nm] | Newton-meter |
| `tool_wear_min` | Tool wear [min] | minutes |
| `type` | Type | categorical: L / M / H |

**Columns explicitly excluded, and why:**
- `UDI`, `Product ID` — row identifiers, no generalizable signal.
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure sub-mode flags) — **dropped as target leakage**. These are components of the target definition itself: their logical OR matches `Machine failure` in 99.73% of rows (verified by direct computation, not assumed from documentation). They record whether a specific failure mechanism occurred and are not available as legitimate pre-failure input at inference time.

## ML methodology

**Split**: stratified 60/20/20 train/validation/test, `random_state=42`, applied via two sequential `train_test_split` calls. Preprocessing (StandardScaler + OneHotEncoder) is fit only on the training split, inside an sklearn `Pipeline`, so it cannot leak into validation or test.

**Model comparison** (validation set, selected by **PR-AUC**, not accuracy — accuracy is a poor metric here since the dataset is 96.6% negative class):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Dummy baseline | 0.489 | 0.034 | 0.014 | 0.015 | 0.014 |
| Logistic Regression | 0.881 | 0.407 | 0.151 | 0.794 | 0.254 |
| Random Forest | 0.957 | 0.590 | 0.541 | 0.676 | 0.601 |
| XGBoost | 0.965 | 0.777 | 0.658 | 0.735 | 0.694 |
| **HistGradientBoosting (selected)** | **0.970** | **0.817** | **0.904** | **0.691** | **0.783** |

**Calibration** was tested, not assumed. `CalibratedClassifierCV` was tried with both isotonic and sigmoid methods; both made Brier score *worse* than the uncalibrated model (uncalibrated: 0.0124, isotonic: 0.0139, sigmoid: 0.0138) — likely because the validation set has only ~68 positive examples, too few for stable calibration. **The model was kept uncalibrated**, and the application deliberately calls the output a "risk score," not a "probability," as a direct consequence of this result.

**Threshold selection**: a sensitivity analysis was run across FN:FP cost-ratio assumptions (1:1, 2:1, 5:1, 10:1) using a deterministic tie-break rule (minimize expected cost → prefer higher recall on ties → prefer higher threshold on remaining ties). The project selected the **5:1 ratio as an explicit, documented modeling assumption** (missing a failure is assumed 5x costlier than a false alarm) — **not a sourced industry figure** — giving an operating threshold of **0.03**. This produces a single binary decision (`riskScore < 0.03` → `NORMAL`, else `ACTION_REQUIRED`), not a three-level scheme: a separate threshold (0.57) came from a different, mutually exclusive cost assumption (1:1/2:1) and is documented for transparency only — it is never used to classify risk, since no experiment validated combining the two into one policy.

**Order-based robustness check** (not a temporal validation test): a second experiment split by row order (`UDI`) instead of randomly, at matching proportions. Performance dropped (PR-AUC 0.817 → 0.691). `UDI` is documented as a row identifier, not a temporal sequence, and AI4I's rows are independently simulated snapshots rather than one machine's continuous trajectory — so **this should not be interpreted as evidence about temporal generalization or "temporal leakage."** It indicates sensitivity to the dataset's ordering/distribution, and is reported as an open, unresolved finding rather than dismissed.

**Threshold stability**: a bootstrap check (1,000 resamples of the validation set only — test set never touched) shows the selected threshold is **not highly stable**: only 37% of resamples reproduce exactly 0.03, the 90% interval spans [0.03, 0.57], and 5.8% of resamples select 0.57 (the *other* cost ratio's threshold) outright. This is expected given the validation set has only ~68 positive examples, but it means **0.03 should be read as "the threshold this stated methodology selects on this data," not as a precisely stable operating point** — the underlying decision boundary near the low-probability region is noisy. The threshold is kept as-is (changing it post hoc based on this check would itself be a form of test-set-blind cherry-picking); the instability is disclosed instead.

**Final test-set metrics** (test set touched exactly once, after every modeling decision was locked on validation data):

- ROC-AUC: 0.980, PR-AUC: 0.808
- Precision: 49.6%, Recall: 82.4%, F1: 0.619
- Confusion matrix: TN=1875, FP=57, FN=12, TP=56
- Recall 95% bootstrap CI: [72.6%, 91.2%] — precision 95% CI: [40.7%, 58.6%] (wide, because the test set has only 68 positive examples — reported explicitly rather than presenting the point estimate as more precise than it is)

**Permutation importance** (sanity check, not a causal claim): Air temperature and Torque dominate, followed by Rotational speed, Tool wear, Process temperature; `Type` contributes almost nothing. This is consistent with AI4I's documented synthetic failure-mode formulas (e.g. HDF depends on temperature difference + rotational speed; PWF on torque×speed).

## API

**`POST /api/predictions`**
```json
// Request
{
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500.0,
  "torque_nm": 40.0,
  "tool_wear_min": 10.0,
  "type": "M"
}
// Response (201)
{
  "id": "...",
  "prediction": { "riskScore": 0.0347, "riskLevel": "ACTION_REQUIRED" },
  "modelVersion": "ai4i-hgb-v1-20260830",
  "predictedAt": "2026-08-30T..."
}
```

**`GET /api/predictions?page=1&limit=20`** — paginated history: `{records, page, limit, totalCount, totalPages}`

**`GET /api/predictions/:id`**, **`DELETE /api/predictions/:id`**

**`DELETE /api/predictions`** — bulk reset, requires `X-Admin-Key` header matching the `ADMIN_KEY` environment variable (see Security below).

**Validation**: Express checks field presence/type (shape). Flask independently checks numeric ranges derived from the model's actual training data (e.g. air temperature must fall within the range the model has training evidence for) — this range check can only live in the ML service, since only it knows what the model has seen.

**Error handling**: distinct codes for `VALIDATION_FAILED` (422), `ML_SERVICE_UNAVAILABLE` (503), `ML_TIMEOUT` (504), `ML_SERVICE_ERROR` (502/503), `MALFORMED_ML_RESPONSE` (502), `PERSISTENCE_FAILED` (503, distinguished from ML failure — the prediction succeeded but couldn't be saved). No internal stack traces or file paths are ever returned to the client.

**Health checks**: Flask exposes `/health/live` (process alive) and `/health/ready` (model artifact loaded, metadata valid, one real test inference already succeeded) separately, so a broken model is caught at startup, not on the first real request.

## Database

```js
Prediction {
  input: { airTemperatureK, processTemperatureK, rotationalSpeedRpm, torqueNm, toolWearMin, type },
  result: { riskScore, riskLevel },       // verbatim from the ML service, never recomputed by Express
  modelVersion,                            // every record traceable to the model that produced it
  predictedAt,
  createdAt
}
```
Indexed on `createdAt` (history sort) and `modelVersion + createdAt` (traceability queries). There is deliberately **no "actual failure" field** — the application has no source of real-world observed outcomes, and adding one would misrepresent a placeholder as ground truth.

## Security

**In scope**: the one destructive endpoint (`DELETE /api/predictions`, bulk reset) is protected by a shared secret (`ADMIN_KEY` env var, sent as `X-Admin-Key` header). CORS is restricted to a configured origin (`CLIENT_ORIGIN`), not left open to all origins. The Flask ML service is never exposed directly to the browser — Express is the only public boundary.

**Explicitly out of scope, and why**: full user authentication (JWT/accounts/sessions) was not implemented. This is a single-user portfolio demo with no per-user data and no sensitive data beyond the shared admin secret; adding a full auth subsystem would add real complexity (registration, password reset, session handling) without a corresponding real requirement. A genuine multi-user deployment would need it.

## Environment variables

**Backend** (`backend/.env`, see `backend/.env.example`):
| Variable | Required | Purpose |
|---|---|---|
| `MONGO_URI` | Yes | MongoDB Atlas connection string |
| `ML_SERVICE_URL` | Yes (defaults to `http://localhost:5001`) | Flask service address |
| `ADMIN_KEY` | No | Enables bulk-delete; bulk-delete is disabled if unset |
| `CLIENT_ORIGIN` | Yes for production (defaults to `http://localhost:3000`) | CORS allowlist |
| `PORT` | No (defaults to 5000) | Express port |

**Client** (`client/.env`, see `client/.env.example`):
| Variable | Required | Purpose |
|---|---|---|
| `REACT_APP_API_URL` | Yes for production (defaults to `http://localhost:5000`) | Express API base URL |

No secrets are committed to this repository (verified via repository-wide scan — see Testing).

## Local setup

```bash
# ML service
cd backend/ml-service
pip install -r requirements.txt
python app.py            # runs on :5001

# Backend
cd backend
npm install
cp .env.example .env     # fill in MONGO_URI
npm start                # runs on :5000

# Frontend
cd client
npm install
cp .env.example .env
npm start                # runs on :3000
```

## Deployment

**Local verification only, at the time of writing.** The architecture above (React/Express/Flask/MongoDB) has been verified end-to-end in a local/sandbox environment, including live HTTP tests of Flask, real (non-mocked) Express→Flask calls, and a production React build. **It has not yet been redeployed** to the project's hosting (Render/Vercel) — the previously deployed instances still reflect an earlier version of this system until redeployed with the current code and environment variables above.

## Testing

- **Flask**: 17 tests (unit + validation + error-handling), all passing, run from the actual served location.
- **Express**: 15 tests covering validation, ML failure modes (unavailable/timeout/malformed response), persistence failure handling, pagination, and the admin-key-protected delete. These test Express's request handling and orchestration logic against a **mocked** Mongoose model layer — **database integration against a real MongoDB instance remains to be verified**; a real MongoDB binary wasn't reachable in the development sandbox used to build this.
- **Frontend**: production build verified to compile successfully.
- **End-to-end**: a real (non-mocked) Express instance was run against a real (non-mocked) Flask instance, confirming: successful predictions reach Flask and return a real score; validation failures never reach Flask; killing Flask mid-test correctly returns 503; a missing database correctly returns a distinct `PERSISTENCE_FAILED` error rather than being conflated with an ML failure.

## Known limitations

- Trained on a public **synthetic/simulated** benchmark dataset, not real industrial sensor data — treat results as a methodology demonstration, not a production performance claim.
- No real-world failure horizon exists in the source data; the system reports a point-in-time risk score, not a "fails within N days" forecast.
- The order-based robustness check shows a real, unexplained performance drop under UDI-ordering — not evidence of temporal leakage (UDI isn't a timestamp), but not fully explained either.
- The selected threshold (0.03) is not highly stable under bootstrap resampling of the validation set — see Threshold stability above.
- Database integration tests are currently mocked, not run against a real MongoDB instance.
- No user authentication; single shared secret protects the one destructive endpoint.
