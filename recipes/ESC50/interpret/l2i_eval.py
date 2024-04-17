#!/usr/bin/python3
"""This recipe to train PIQ to interepret audio classifiers.

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
"""
import os
import sys
from os import makedirs

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
from esc50_prepare import prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from train_l2i import InterpreterESC50Brain
from wham_prepare import WHAMDataset, combine_batches

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats

eps = 1e-10


class EvalL2I(InterpreterESC50Brain):
    def interpret_computation_steps(self, predictions, f_I):
        """computation steps to get the interpretation spectrogram"""
        # get the nmf activations
        psi_out = self.modules.psi(f_I)

        if isinstance(psi_out, tuple):
            psi_out = psi_out[0]
            psi_out = psi_out.squeeze(1).permute(0, 2, 1)

        # cut the length of psi in case necessary
        # psi_out = psi_out[:, :, : X_stft_power.shape[1]]

        pred_cl = torch.argmax(predictions, dim=1)[0].item()

        nmf_dictionary = self.hparams.nmf_decoder.return_W()

        # computes time activations per component
        # FROM NOW ON WE FOLLOW THE PAPER'S NOTATION
        psi_out = psi_out.squeeze()
        z = self.modules.theta.hard_att(psi_out).squeeze()
        theta_c_w = self.modules.theta.classifier[0].weight[pred_cl]

        # some might be negative, relevance of component
        r_c_x = theta_c_w * z / torch.abs(theta_c_w * z).max()
        # define selected components by thresholding
        L = (
            torch.arange(r_c_x.shape[0])
            .to(r_c_x.device)[r_c_x > self.hparams.relevance_th]
            .tolist()
        )

        X_withselected = nmf_dictionary[:, L] @ psi_out[L, :]
        Xhat = nmf_dictionary @ psi_out

        # need the eps for the denominator
        eps = 1e-10
        X_hat = X_withselected / (Xhat + eps)

        return X_hat.transpose(-1, -2)


def l2i_pretrained(hparams, run_opts):
    l2i_brain = EvalL2I(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    @torch.no_grad()
    def explain_fn(ft, _, model):
        predictions, temp = model(ft)
        mask = l2i_brain.interpret_computation_steps(predictions, temp)[
            None, None
        ]

        return mask

    return explain_fn
