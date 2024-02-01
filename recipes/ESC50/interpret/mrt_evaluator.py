from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from tqdm import tqdm
import scipy
import torch
import sys
import os

m = sys.argv[1]
BASE_FOLDER = "."
PATHS = [m]
PATHS += [f"{m}_mrt_layer{i}" for i in range(1, 7)]

PATHS = [Path(BASE_FOLDER).joinpath(f"qualitative_{p}") for p in PATHS]

if __name__ == "__main__":
    metrics = {
            "rank": [],
            "rankabs": [],
            "ssim": []
            }
    for p in PATHS[1:]:
        temp = {
            "rank": [],
            "rankabs": [],
            "ssim": []
            }
        assert os.path.exists(p), f"{p} does not exist."

        skipped = 0
        for el in tqdm(sorted(p.iterdir()), desc=f"Running on {str(p)}..."):
            if "csv" in str(el.name):
                continue

            original = torch.load(PATHS[0].joinpath(el.name, "interpretation.pt")).detach().cpu().squeeze().numpy()
            mrt = torch.load(p.joinpath(el.name, "interpretation.pt")).detach().cpu().squeeze().numpy()

            if original.std() == 0 or mrt.std() == 0:
                skipped += 1
                continue

            temp["rank"].append(scipy.stats.spearmanr(original.flatten(), mrt.flatten()).statistic)
            temp["rankabs"].append(scipy.stats.spearmanr(original.flatten(), mrt.flatten(), alternative="greater").statistic)
            temp["ssim"].append(ssim(original, mrt, data_range=mrt.max() - mrt.min()))


        print(f"{str(p)} completed! -- Total skipped={skipped}")
        for k in temp:
            temp[k] = sum(temp[k]) / len(temp[k])
        print(temp)

