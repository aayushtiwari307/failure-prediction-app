# Predictive Maintenance Risk Assessment System

A full-stack application that estimates machine failure risk from operating sensor readings, trained and evaluated on a public benchmark dataset. Built as a portfolio project demonstrating an end-to-end ML + web engineering pipeline: data leakage analysis, model comparison, calibration evaluation, cost-sensitive threshold selection, a validated model-serving API, and persistence — with honest, documented limitations rather than inflated claims.

**Live app:** https://failure-prediction-app.vercel.app
**Backend API:** https://failure-prediction-backend.onrender.com
**ML service:** https://failure-prediction-ml.onrender.com

> Free-tier hosting note: the backend and ML services spin down after inactivity and may take ~50 seconds to respond on the first request after idling.

---

## What this is (and isn't)

This project predicts failure risk from the **AI4I 2020 Predictive Maintenance Dataset** (UCI), a public, synthetic benchmark dataset — **not real industrial telemetry**. The risk score is an uncalibrated model output, not a guaranteed real-world probability of failure. Every claim below is backed by an experiment; where something is a limitation, it's stated as one.

---

## Architecture

```
React (client) → Express (validation, orchestration) → Flask (ML + risk policy) → HistGradientBoosting model
                        ↓
                    MongoDB (persistence)
```

- **React** — sends named sensor fields, contains zero threshold logic, renders only what the backend returns
- **Express** — validates request shape/types, calls the ML service, persists results, never computes risk itself
- **Flask** — loads the frozen model artifact, does ML-specific range validation, and is the single source of truth for risk classification
- **MongoDB** — stores each input, result, model version, and timestamp

Flask is never public-facing; Express is the only public API boundary.

---

## API

**Express (public)**
| Method | Route | Description |
|---|---|---|
| POST | `/api/predictions` | Validate → call ML service → persist → return result |
| GET | `/api/predictions?page=&limit=` | Paginated prediction history |
| GET | `/api/predictions/:id` | Fetch one prediction |
| DELETE | `/api/predictions/:id` | Delete one prediction |
| DELETE | `/api/predictions` | Bulk reset (requires `X-Admin-Key` header) |

**Flask (internal only)**
| Method | Route |
|---|---|
| GET | `/health/live`, `/health/ready` |
| GET | `/model-info` |
| POST | `/predict` |

**Request body** (`POST /api/predictions`):
```json
{
  "air_temperature_k": 300,
  "process_temperature_k": 310,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40,
  "tool_wear_min": 100,
  "type": "M"
}
```

**Response:**
```json
{
  "modelVersion": "ai4i-hgb-v1-20260830",
  "riskScore": 0.0001,
  "riskLevel": "NORMAL",
  "thresholdPolicy": { "threshold": 0.03, "costRatioAssumed": "5:1" },
  "predictedAt": "2026-09-01T10:39:30.000Z"
}
```

Risk policy is **binary**: `riskScore < 0.03` → `NORMAL`, otherwise → `ACTION_REQUIRED`.

---

## ML methodology

**Pipeline:** AI4I 2020 → leakage analysis → drop failure-subtype columns (verified 99.73% target overlap, not assumed) and identifier columns → stratified 60/20/20 split (`random_state=42`) → `sklearn` preprocessing pipeline (fit train-only) → 5-model comparison by validation PR-AUC → calibration check → cost-sensitive threshold selection → test set touched exactly once → bootstrap confidence intervals → frozen artifact.

**Model comparison (validation PR-AUC):**
| Model | PR-AUC |
|---|---|
| Dummy | 0.034 |
| Logistic Regression | 0.407 |
| Random Forest | 0.590 |
| XGBoost | 0.777 |
| **HistGradientBoosting (selected)** | **0.817** |

The model was selected empirically based on observed validation performance under a fixed protocol — not for sophistication.

**Calibration:** Tested isotonic and sigmoid calibration; both were rejected because the uncalibrated model had a lower Brier score (0.0124 vs 0.0139 / 0.0138). The model output is explicitly *not* a calibrated probability.

**Threshold selection:** Cost assumption FN:FP = 5:1 (an explicit modeling assumption, not a sourced industry figure), minimizing expected cost with a deterministic tie-break. Selected threshold: **0.03**.

**Bootstrap robustness check (n=1000, validation set):** the threshold is not highly stable — only 37% of resamples reproduce exactly 0.03, with a 90% interval of [0.03, 0.57]. Disclosed as a limitation; the threshold was kept as-is rather than cherry-picked post hoc against the test set.

### Final locked metrics (test set, touched once)

| Metric | Value |
|---|---|
| ROC-AUC | 0.980 |
| PR-AUC | 0.808 |
| Precision | 49.6% |
| Recall | 82.4% |
| F1 | 0.619 |

Confusion matrix: TN 1875, FP 57, FN 12, TP 56. Bootstrap 95% CIs: Recall [72.6%, 91.2%], Precision [40.7%, 58.6%].

**Feature importance (permutation, sanity check only, not causal):** Air temperature and torque dominate, followed by rotational speed and tool wear; machine type contributes negligibly.

---

## Known limitations

- Dataset is a public, simulated benchmark — not production telemetry
- `riskScore` is not a calibrated probability (calibration was tested and rejected)
- The 0.03 threshold is not highly stable under bootstrap resampling
- An order-based robustness check showed a real, unresolved PR-AUC drop (0.817 → ~0.70) when splitting by row order instead of stratified sampling — flagged as an open finding, not evidence of temporal leakage (the dataset's row order is not a timestamp)
- Input validation bounds are training-data-evidence ranges, not physical or industry safety limits
- No full user authentication — single-user portfolio scope, with one destructive endpoint protected by a shared admin key instead
- No live MongoDB integration test in CI (verified manually against MongoDB Atlas in production instead)

---

## Testing

- Flask/ML: 17/17 unit tests passing, plus live HTTP smoke tests of valid/invalid/malformed requests and health endpoints
- Express: 17/17 tests passing (mocked Mongoose model + real Mongoose schema validation)
- Frontend: production build compiles cleanly
- Full live end-to-end verification: real request through React → Express → Flask → MongoDB Atlas and back, confirmed in production

---

## Tech stack

React · Express · Flask · scikit-learn (HistGradientBoosting) · MongoDB Atlas · deployed on Vercel (frontend) and Render (backend + ML service)
