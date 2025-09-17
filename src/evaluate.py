import argparse
import json
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

# opsional: logging ke MLflow (boleh hapus jika belum perlu)
try:
    import mlflow
except ImportError:
    mlflow = None

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "island", "sex"]

def safe_write_json(path: str, payload: dict):
    # pastikan folder ada
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def main(test_csv: str, model_path: str, report_out: str):
    # Tulis stub lebih dulu supaya DVC selalu melihat file ada
    safe_write_json(report_out, {"status": "running"})

    df = pd.read_csv(test_csv)
    y_true = df["species"]
    X = df[FEATURES]

    clf = joblib.load(model_path)
    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)

    out = {
        "status": "ok",
        "accuracy": acc,
        "f1_macro": f1m,
        "classification_report": report
    }
    safe_write_json(report_out, out)

    # (opsional) log ke MLflow jika terpasang
    if mlflow is not None:
        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("penguins_rf")
        with mlflow.start_run(run_name="eval"):
            mlflow.log_metrics({"accuracy": acc, "f1_macro": f1m})
            mlflow.log_artifact(report_out)
            mlflow.set_tags({"stage": "eval"})

    # print ringkas ke stdout (biar keliatan di log CI)
    print(json.dumps({"accuracy": acc, "f1_macro": f1m}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    main(args.test, args.model, args.report)
