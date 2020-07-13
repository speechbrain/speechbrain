import csv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from librispeech_prepare import prepare_librispeech  # noqa E402


data_folder = os.path.abspath(sys.argv[-1])

csv_list = [
    "train-clean-960",
    # 'train-clean-360'
    # "train-clean-100",
    # 'train-other-500'
]

sentenc_list = []

for c in csv_list:
    print("extracting sentences form {}...".format(c))
    prepare_librispeech(
        data_folder=data_folder,
        splits=[c, "dev-clean", "test-clean"],
        save_folder=data_folder,
    )

    with open(os.path.join(data_folder, c + ".csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sentenc_list.append(row["wrd"])
    print("done!")

print("found {} sentences..".format(len(sentenc_list)))

print("writing sentences to data folder...")

with open(os.path.join(data_folder, "transcript-960.txt"), "w") as f:
    f.writelines("%s\n" % sent for sent in sentenc_list)
