
import argparse
import os
import pandas as pd

def main(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}, shape={df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.url, args.out)
