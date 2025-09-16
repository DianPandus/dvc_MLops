
# MLOps: MLflow + DVC + GitHub Actions (Penguins Classification)

This repo demonstrates an end‑to‑end MLOps workflow:

- **DVC** to build and reproduce the data & modeling pipeline
- **MLflow** to track experiments and register the trained model
- **GitHub Actions** CI to run `dvc repro` and upload artifacts on every push
- **Public dataset**: [Palmer Penguins](https://github.com/allisonhorst/palmerpenguins)

## Quickstart

```bash
# 1) Create env & install deps
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Initialize DVC (once)
dvc init

# 3) Reproduce full pipeline
dvc repro

# 4) See metrics
cat reports/metrics.json

# 5) View MLflow runs locally
mlflow ui --backend-store-uri mlruns
```

## Pipeline stages (DVC)

1. **get_data** → downloads `penguins.csv` into `data/raw/`
2. **split_data** → creates `data/processed/train.csv` & `test.csv`
3. **train** → trains a scikit‑learn pipeline, logs to MLflow, saves `models/model.joblib`
4. **eval** → evaluates model and writes `reports/metrics.json`

Re-run everything with `dvc repro` after changing code or `params.yaml`.

## Report Template

See `REPORT.md` for the experiment write-up you can submit.
