import json
import pandas as pd

csv = pd.read_csv("dev.csv")
csv_dict = {}
for row in range(len(csv)):
    csv_dict[csv.iloc[row]["ID"]] = {
        "wav": csv.iloc[row][" wav"].replace("$data_folder", "{data_root}"),
        "length": csv.iloc[row][" duration"],
        "spk_id": csv.iloc[row][" spk_id"],
        "ali": csv.iloc[row][" ali"].replace("$data_folder", "{data_root}"),
        "phn": " ".join([x.lower().strip(" ") for x in csv.iloc[row][" phn"].split(" ")]), "char": " ".join([x.lower().strip(" ") for x in csv.iloc[row]["char"].split(" ")])

    }



with open("dev.json", "w") as f:
    json.dump(csv_dict, f, indent=4)