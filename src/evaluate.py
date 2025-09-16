
import argparse
import json
import pandas as pd
import joblib
import os
import mlflow
from sklearn.metrics import accuracy_score, f1_score, classification_report

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "island", "sex"]


def main(test_csv: str, model_path: str, report_out: str):
    df = pd.read_csv(test_csv)
    y_true = df["species"]
    X = df[FEATURES]
    clf = joblib.load(model_path)
    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)

    out = {"accuracy": acc, "f1_macro": f1m, "classification_report": report}

    # pastikan folder ada
    os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(out, f, indent=2)

    # === MLflow logging (run eval terpisah) ===
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("penguins_rf")
    with mlflow.start_run(run_name="eval"):
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1m})
        mlflow.log_artifact(report_out)
        mlflow.set_tags({"stage": "eval"})

    print(json.dumps({"accuracy": acc, "f1_macro": f1m}, indent=2))

