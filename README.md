# Cloud AI Advisory Layer (No API)

This repository contains the cloud-only advisory intelligence components for predictive maintenance.

## Scope

Cloud logic remains advisory-only and includes:

- Remaining Useful Life (RUL) prediction
- Failure probability prediction
- Fault explanation
- Maintenance recommendation

No real-time control or physical actuation is implemented here.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

Generate synthetic training data and train model artifacts:

```bash
python scripts/run_full_pipeline.py --output-dir . --rows 1000 --seed 42
```

Artifacts produced:

- `cloud_health_history.csv`
- `rul_model.pkl`
- `failure_model.pkl`

## Validate everything end-to-end (without API)

```bash
python scripts/test_everything.py
```

This runs the full local cloud pipeline:

1. Generate Section-6-style synthetic history
2. Train RUL and failure models
3. Run local inference using the trained models
4. Run explanation + recommendation engines and validate outputs

## Input contract used by local inference

The local advisory pipeline uses Section-6 style fields represented by `CloudInput` in `cloud_ai/schemas.py`.

## Tests

```bash
python -m compileall cloud_ai tests scripts
python -m unittest discover -s tests -v
```
