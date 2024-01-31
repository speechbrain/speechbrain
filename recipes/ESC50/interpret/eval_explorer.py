from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from pathlib import Path
import random
import shutil
import torch
import sys
import os

selection_seed = 1234
random.seed(selection_seed)

BASE_FOLDER = "/mnt/data/icml2024_exps/"
EVAL_LIST = [
        "ao",
        "finetune_16_cth_0.7",
        "finetune_gw_4",
        "finetune_gw_16",
        "finetune_gw_32",
        "finetune_gw_128",
        "l2i_0.2_OOD",
        "l2i_0.4_OOD",
        "l2i_0.6_OOD",
        "l2i_0.8_OOD",
        ]
EVAL_LIST = [Path(BASE_FOLDER).joinpath(f"qualitative_{e}") for e in EVAL_LIST]  # pre-prend base folder

KEPT_PATH = Path("user_study") # path for all the samples we keep


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

@torch.no_grad()
def selection(id_: str) -> str:
    choice_folder = Path("choice")
    os.makedirs(choice_folder, exist_ok=True)

    paths = [e.joinpath(id_) for e in EVAL_LIST]
    images = []
    labels = []
    for p in paths:
        assert os.path.exists(p), f"{p} does not exist. It is likely that something was wrong during eval."
        l = "_".join(str(p.parent).split("/")[-1].split("_")[1:])

        int_ = torch.load(p.joinpath("interpretation.pt")).cpu().squeeze().t()
        sample = torch.load(p.joinpath("x_logpower.pt"))[:, :, :int_.shape[-1], :].cpu().squeeze().t()

        images.append(sample * int_)

        labels.append(l)
        shutil.copyfile(
                src=p.joinpath(f"{id_}_int.wav"),
                dst=choice_folder.joinpath(f"{l}.wav")
                )

    images.append(sample)
    labels.append("sample")
    shutil.copyfile(
            p.joinpath(f"{id_}_original.wav"),
            choice_folder.joinpath(f"original.wav")
            )

    fig = plt.figure()
    grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(1, len(EVAL_LIST) + 1),  # creates 2x2 grid of axes
            axes_pad=0.1,  # pad between axes in inch.
            )

    for ax, im, l in zip(grid, images, labels):
        ax.imshow(im, origin="lower")
        ax.set_title(l)

    plt.tight_layout()
    plt.savefig(choice_folder.joinpath("choice.png"))

    y = None
    while not (y in ["y", "n", "q"]):
        y = input("Keep this sample? [y/n/q] ")

    return y

def keep_sample(id_: str) -> None:
    paths = [e.joinpath(id_) for e in EVAL_LIST]
    labels = ["_".join(str(p.parent).split("/")[-1].split("_")[1:]) for p in paths]
    for p, l in zip(paths, labels):
        dst_dir = KEPT_PATH.joinpath(id_, l)
        os.makedirs(dst_dir, exist_ok=True)
        copytree(p, dst_dir)


if __name__ == "__main__":
    os.makedirs(KEPT_PATH, exist_ok=True)

    for e in EVAL_LIST:
        assert os.path.exists(e), f"{e} does not exist. Check your EVAL list."

    uttid = [e.name for e in sorted(EVAL_LIST[0].iterdir())]
    random.shuffle(uttid)

    for id_ in uttid:
        print(f"Prompting sample... {id_}")

        keep = selection(id_)

        if keep == "q":
            exit(0)
        elif keep == "y":
            keep_sample(id_)

