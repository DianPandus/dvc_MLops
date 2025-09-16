
import argparse
import os
import json
import pandas as pd
import joblib

import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

NUM_FEATS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CAT_FEATS = ["island", "sex"]

def load_train(train_csv: str):
    df = pd.read_csv(train_csv)
    y = df["species"]
    X = df[NUM_FEATS + CAT_FEATS]
    return X, y

def build_model(n_estimators: int, max_depth, min_samples_split: int, random_state: int):
    num_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_proc, NUM_FEATS),
        ("cat", cat_proc, CAT_FEATS)
    ])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def main(train_csv: str, model_out: str, params_json: str, experiment: str = "penguins_rf"):
    with open(params_json) as f:
        params = json.load(f)

    X, y = load_train(train_csv)
    pipe = build_model(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        min_samples_split=params["model"]["min_samples_split"],
        random_state=params["model"]["random_state"],
    )

    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment(experiment)

    with mlflow.start_run():
        mlflow.log_params(params["model"])
        pipe.fit(X, y)
        joblib.dump(pipe, model_out)
        mlflow.log_artifact(model_out, artifact_path="model")
        mlflow.sklearn.log_model(pipe, artifact_path="sk_model", registered_model_name="penguins_rf")

        # Log feature names (after OHE handled by pipeline it's dynamic; we record raw)
        mlflow.log_dict({"num_feats": NUM_FEATS, "cat_feats": CAT_FEATS}, "features.json")

    print(f"Saved model to {model_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params_json", required=True)
    args = ap.parse_args()

    # Convert params.yaml to json once (DVC will pass json for reproducibility)
    main(args.train, args.model, args.params_json)
