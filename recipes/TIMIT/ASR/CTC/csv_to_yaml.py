import pandas as pd
import argparse
import yaml
from pathlib import Path
import os

parser = argparse.ArgumentParser("Converting csv to yaml format")
parser.add_argument(
    "--in_csv", default="/media/sam/HitachiX360/TIMIT/train.csv"
)
parser.add_argument("--out_yml", required=False, default="")

args = parser.parse_args()

csv = pd.read_csv(args.in_csv)

y = {}
for i in range(len(csv)):
    line = csv.iloc[i]
    y[line["ID"]] = {
        "wav": line["wav"],
        "spk_id": line["spk_id"],
        "phn": line["phn"].splineit(" "),
        "length": int(line["duration"] * 16000),
    }

y = {"examples": y}

if not len(args.out_yml):
    out_yml = os.path.join(
        Path(args.in_csv).parent, Path(args.in_csv).stem + ".yaml"
    )
else:
    out_yml = args.out_yml

with open(out_yml, "w") as f:
    yaml.dump(y, f, indent=4)
