
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_csv: str, outdir: str, test_size: float, seed: int):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # Drop rows with all‑NaN in critical columns
    df = df.dropna(subset=["species"])

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["species"])
    train_path = os.path.join(outdir, "train.csv")
    test_path = os.path.join(outdir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--test_size", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    main(args.input, args.outdir, args.test_size, args.seed)
