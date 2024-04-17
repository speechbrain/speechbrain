from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np
import torch

fsize = 11
tsize = 18
tdir = "in"
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use("default")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = fsize
plt.rcParams["legend.fontsize"] = tsize
plt.rcParams["xtick.direction"] = tdir
plt.rcParams["ytick.direction"] = tdir
plt.rcParams["xtick.major.size"] = major
plt.rcParams["xtick.minor.size"] = minor
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["axes.linewidth"] = lwidth
plt.rcParams["legend.handlelength"] = lhandle

METHODS = ["ao", "ft_4", "ft_16"]
METHODS = [Path(f"qualitative_{m}") for m in METHODS]

if __name__ == "__main__":

    selection = 10
    for s in METHODS:
        for idx, el in enumerate(sorted(s.iterdir())):
            if selection != idx:
                continue

            if idx > selection:
                break

            int_ = torch.load(el.joinpath("interpretation.pt")).detach().cpu().squeeze().numpy().T
            sample = torch.load(el.joinpath("x_logpower.pt")).detach().cpu().squeeze().numpy().T

            plt.imshow(int_, origin="lower", cmap="inferno")
            plt.savefig(f"mrt_viz/{s.name}_mask.pdf")

            plt.imshow(int_ * sample, origin="lower", cmap="inferno")
            plt.savefig(f"mrt_viz/{s.name}.pdf")

            plt.imshow(sample, origin="lower", cmap="inferno")
            plt.savefig(f"mrt_viz/original.pdf")


