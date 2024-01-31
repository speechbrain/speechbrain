from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from tqdm import tqdm
import scipy
import torch
import os

BASE_FOLDER = "."
PATHS = ["ao"]
PATHS += [f"mrt_layer{i}" for i in range(1, 7)]

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

        for el in tqdm(sorted(p.iterdir()), desc=f"Running on {str(p)}..."):
            if "csv" in str(el.name):
                continue

            original = torch.load(PATHS[0].joinpath(el.name, "interpretation.pt")).cpu().squeeze().numpy()
            mrt = torch.load(p.joinpath(el.name, "interpretation.pt")).cpu().squeeze().numpy()

            temp["rank"].append(scipy.stats.spearmanr(original.flatten(), mrt.flatten()).statistic)
            temp["rankabs"].append(scipy.stats.spearmanr(original.flatten(), mrt.flatten(), alternative="greater").statistic)
            temp["ssim"].append(ssim(original, mrt, data_range=mrt.max() - mrt.min()))


        print(f"{str(p)} completed!")
        for k in temp:
            temp[k] = sum(temp[k]) / len(temp[k])
        print(temp)

