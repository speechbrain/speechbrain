import json
import pandas as pd
import os
from pathlib import Path


def csv_to_json(in_csv, out_yaml=None):
    csv = pd.read_csv(in_csv)

    y = {}
    for i in range(len(csv)):
        line = csv.iloc[i]
        y[line["ID"]] = {
            "wav": line["wav"],
            "spk_id": line["spk_id"],
            "phn": line["phn"],
            "length": int(line["duration"] * 16000),
        }

    if out_yaml is None:
        out_yml = os.path.join(Path(in_csv).parent, Path(in_csv).stem + ".json")

    with open(out_yml, "w") as f:
        json.dump(y, f, indent=4)
