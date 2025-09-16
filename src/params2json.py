
import yaml, json, argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="params.yaml")
    ap.add_argument("--outfile", default="params.json")
    a = ap.parse_args()
    with open(a.infile) as f:
        y = yaml.safe_load(f)
    with open(a.outfile, "w") as f:
        json.dump(y, f)
    print(f"Wrote {a.outfile}")
