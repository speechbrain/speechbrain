#!/usr/bin/python3
"""This recipe to train PIQ to interepret audio classifiers.

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
"""
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from speechbrain.utils.metric_stats import MetricStats
from wham_prepare import WHAMDataset, combine_batches
from os import makedirs
import torch.nn.functional as F
from speechbrain.processing.NMF import spectral_phase
import matplotlib.pyplot as plt

eps = 1e-10


def opt_single_mask(ft, model, EP=1000):
    mask = torch.nn.Parameter(
        torch.ones(ft.shape, device="cuda") * 0.5, requires_grad=True
    )
    opt = torch.optim.Adam([mask], lr=1e-3)

    for e in range(EP):
        opt.zero_grad()

        xhat = model(ft)
        argmax = xhat.argmax(-1)

        xhat = model(ft * torch.sigmoid(mask)).squeeze()
        xhat2 = model(ft * (1 - torch.sigmoid(mask))).squeeze()

        loss = -xhat[argmax] + xhat2[argmax] - xhat2.mean() + mask.abs().mean()

        loss.backward()

        opt.step()

    return mask.detach()


@torch.no_grad()
def interpret_pretrained(interpreter):
    interpreter = interpreter.eval()

    @torch.no_grad()
    def interpret_ao(x, _, model):
        predictions, fI = model(x)
        pred_cl = predictions.argmax(1)

        # print([f.shape for f in fI])
        if not model.cnn14:
            temp = interpreter.decoder(fI[1])
        else:
            temp = interpreter(fI)
        temp = torch.sigmoid(temp)
        temp = temp[:, :, : x.shape[2], : x.shape[3]]

        return temp

    return interpret_ao


def all_onesmask(x, _, model):
    return torch.ones(x.shape, device=x.device)


