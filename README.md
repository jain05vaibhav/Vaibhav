# Cloud AI Advisory Layer (No API)

This repository implements the **cloud-only advisory intelligence layer** for predictive maintenance.

It is designed to do only analytics/advisory work:

- Remaining Useful Life (RUL) prediction
- Failure probability prediction (next 7 days)
- Fault explanation (rule-based)
- Maintenance recommendation (policy-based)

It **does not** do real-time safety control or physical actuation.

---

## 1) What this project is (and is not)

### ✅ What it does

- Trains 2 cloud ML models from Section-6-style processed vehicle health data:
  - RUL regressor (`rul_model.pkl`)
  - Failure classifier (`failure_model.pkl`)
- Runs local end-to-end cloud advisory inference (no web server required).
- Produces explainable outputs (fault cause + recommendation fields).
- Includes tests for data generation, training, advisory logic, and full pipeline.

### ❌ What it does not do

- No ESP32/fog/hardware integration in this repo.
- No direct vehicle control.
- No built-in deployment to cloud infra in this repo.

---

## 2) Project structure

```text
cloud_ai/
  data_generation.py     # synthetic Section-6-like training data
  rul_model.py           # train RUL model
  failure_model.py       # train failure probability model
  explanation.py         # rule-based fault explanation
  recommendation.py      # policy-based recommendation
  schemas.py             # input contract model (CloudInput)
  pipeline.py            # one-call generate + train pipeline
scripts/
  run_full_pipeline.py   # generate dataset + train both models
  test_everything.py     # complete local validation flow
tests/
  test_data_generation.py
  test_advisory_logic.py
  test_end_to_end.py
  test_pipeline.py
```

---

## 3) Prerequisites

- Python 3.10+
- Linux/macOS/WSL recommended

Check Python version:

```bash
python --version
```

---

## 4) Setup (fresh machine)

From repository root:

```bash
# 1) (optional but recommended) create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt
```

Expected core dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`

---

## 5) End-to-end run (start to finish)

### Step A — Generate synthetic dataset + train models

```bash
python scripts/run_full_pipeline.py --output-dir . --rows 1000 --seed 42
```

This should create these files in current directory:

- `cloud_health_history.csv`
- `rul_model.pkl`
- `failure_model.pkl`

### Step B — Run complete local cloud validation

```bash
python scripts/test_everything.py
```

This script performs all of the following in order:

1. Generate Section-6-like synthetic data.
2. Train both models.
3. Load trained artifacts.
4. Run local inference for RUL + failure probability.
5. Run explanation engine.
6. Run recommendation engine.
7. Validate output ranges and required output keys.

If successful, the script prints final validated advisory output.

### Step C — Run automated tests

```bash
python -m compileall cloud_ai tests scripts
python -m unittest discover -s tests -v
```

You should see all tests passing (`OK`).

---

### RUL model feature note (as requested)

RUL training/inference uses only these three inputs:

- `engine_rul_pct`
- `brake_rul_pct`
- `battery_rul_pct`

## 6) How to verify each part individually

### 6.1 Data generation only

```bash
python -m cloud_ai.data_generation
```

Verify file exists:

```bash
ls -lh cloud_health_history.csv
```

### 6.2 Train only RUL model

```bash
python -m cloud_ai.rul_model
```

Expected output includes `RUL MAE`, `RUL R2`, and `Saved model to rul_model.pkl`.

### 6.3 Train only failure model

```bash
python -m cloud_ai.failure_model
```

Expected output includes `Failure model AUC` and `Saved model to failure_model.pkl`.

### 6.4 Advisory logic only (unit tests)

```bash
python -m unittest tests.test_advisory_logic -v
```

---

## 7) Using your real car/fleet data (important)

Yes, this can be used for car analytics workflows, **but with conditions**:

1. Your fog/system must provide the same processed feature schema style as `CloudInput` (Section-6 style fields).
2. You must train with representative historical data from your vehicles, not only synthetic data.
3. You should retrain periodically as usage patterns change.
4. Keep this advisory-only; do not use directly for real-time actuation.

### Minimal practical path for real car data

1. Export historical processed records (CSV) with required columns used by the models.
   - For RUL specifically, include: `engine_rul_pct`, `brake_rul_pct`, `battery_rul_pct`, and target `engine_rul_pct_future`.
2. Replace synthetic CSV with your historical CSV (same column names).
3. Train models via:
   - `python -m cloud_ai.rul_model`
   - `python -m cloud_ai.failure_model`
4. Run your local validation flow to verify outputs are sane.
5. Integrate this module into your cloud service layer (outside this repo) that consumes fog Section-6 records.

---

## 8) “Will it work for car na?”

Short answer: **Yes for cloud advisory analytics, not as direct vehicle control software.**

- It will work for **car health prediction/recommendation pipelines** if your incoming data matches expected processed features.
- It is **not** plug-and-play ECU/actuation software.
- For production vehicle usage, you need:
  - real historical training data,
  - proper validation/calibration,
  - monitoring,
  - safe system integration boundaries (fog controls actuation, cloud remains advisory).

---

## 9) Troubleshooting

### `ModuleNotFoundError: cloud_ai`
Run from repository root and/or use:

```bash
export PYTHONPATH=.
```

### Model file missing
Train models first:

```bash
python scripts/run_full_pipeline.py --output-dir . --rows 1000 --seed 42
```

### Tests failing due to missing deps
Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Different metrics each run
Set fixed seed (already supported in scripts) for reproducibility.

---

## 10) Final recommended command sequence (copy-paste)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_pipeline.py --output-dir . --rows 1000 --seed 42
python scripts/test_everything.py
python -m compileall cloud_ai tests scripts
python -m unittest discover -s tests -v
```

If all commands succeed, your cloud advisory layer is fully generated, trained, and verified end-to-end.
