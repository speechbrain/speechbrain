#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gradient_based.py
# Author            : Francesco Paissan <francescopaissan@gmail.com>
# Date              : 07.01.2024
# Last Modified Date: 07.01.2024
# Last Modified By  : Francesco Paissan <francescopaissan@gmail.com>
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import captum

@torch.no_grad()
def save_ints(inp, ints, labels, out_folder=".", fname="viz.png", cbar=True):
    """ Iterate over all the generated interpretations and visualize. """
    subplot_base = 3 * 100 + len(ints) * 10

    line_counter = 1
    for idx in range(len(ints)):
        cplot = line_counter
        plt.subplot(subplot_base + cplot)
        plt.imshow(inp[0].clone().cpu().t(), origin="lower")
        if cbar:
            plt.colorbar()
        plt.title("input")

        cplot = line_counter + 2 if len(ints) > 1 else line_counter + 1
        plt.subplot(subplot_base + cplot)
        plt.subplot(int(subplot_base + cplot))
        plt.imshow(ints[idx].clone().cpu().t(), origin="lower")
        if cbar:
            plt.colorbar()
        plt.title(labels[idx])

        cplot = line_counter + 4 if len(ints) > 1 else line_counter + 2
        plt.subplot(subplot_base + cplot)
        plt.subplot(int(subplot_base + cplot))
        plt.imshow((ints[idx] * inp[0]).clone().cpu().t(), origin="lower")
        if cbar:
            plt.colorbar()
        plt.title("mask in")

        line_counter += 1

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, fname))
    plt.cla(), plt.clf()

@torch.no_grad()
def save_waves(inp, ints, labels, to_wav, out_folder="."):
    """ Iterate over all the generated interpretations and visualize. """

    original = to_wav(inp[0], inp[1])
    original /= original.max()
    torchaudio.save(
            os.path.join(out_folder, "original.wav"),
            original.cpu(),
            sample_rate=16000
            )

    for idx in range(len(ints)):
        int_t = to_wav((inp[0] * ints[idx][0]), inp[1])
        int_t /= int_t.max()

        torchaudio.save(
                os.path.join(out_folder, labels[idx] + ".wav"),
                int_t.cpu(),
                sample_rate=16000
                )

def saliency(X, y, forward_fn, do_norm=True):
    """ Computes standard saliency - gradient of the max logit wrt to the input. """
    X = torch.Tensor(X).to(next(forward_fn.parameters()).device)
    y = torch.Tensor(y).to(next(forward_fn.parameters()).device).long()
    attr = captum.attr.Saliency(forward_fn).attribute(X, target=y)

    if do_norm:
        return attr / attr.max()

    return attr

def smoothgrad(X, y, forward_fn, steps=50, noise_level=0.15, guidance=False):
    """ Runs smoothgrad - gauss noise on input before saliency map. """
    # derive sigma from noise level -- see Fig 3. https://arxiv.org/pdf/1706.03825.pdf
    X = torch.Tensor(X).to(next(forward_fn.parameters()).device)
    y = torch.Tensor(y).to(next(forward_fn.parameters()).device).long()

    # noise_level * std of input NB: we cannot have noise that makes the log arg negative!
    sigma = noise_level * (X.max() - X.min())
    noise = (torch.randn_like(X) * sigma).abs()

    # I tried with captum's NoiseTunnel. it gives problems during preprocessing (see above)
    maps = []
    for _ in range(steps):
        X_perturb = noise + X
        slc = saliency(X_perturb, y, forward_fn)

        maps.append(slc) 

    return torch.stack(maps).mean(dim=0)

def ig(X, y, forward_fn, steps=200):
    """ Computes IG starting from X_baseline to X. """
    X = torch.Tensor(X).to(next(forward_fn.parameters()).device)
    y = torch.Tensor(y).to(next(forward_fn.parameters()).device).long()
    X_baseline = torch.zeros_like(X)

    attr = captum.attr.IntegratedGradients(forward_fn).attribute(
            X,
            baselines=X_baseline,
            target=y,
            internal_batch_size=2,
            ).abs()

    return attr / attr.max()

def guided_backprop(X, y, forward_fn, do_norm=True):
    """ Computes standard saliency - gradient of the max logit wrt to the input. """
    X = torch.Tensor(X).to(next(forward_fn.parameters()).device)
    y = torch.Tensor(y).to(next(forward_fn.parameters()).device).long()
    attr = captum.attr.GuidedBackprop(forward_fn).attribute(X, target=y)

    if do_norm:
        return attr / attr.max()

    return attr

def guided_gradcam(X, y, forward_fn, do_norm=True):
    """ Computes standard saliency - gradient of the max logit wrt to the input. """
    X = torch.Tensor(X).to(next(forward_fn.parameters()).device)
    y = torch.Tensor(y).to(next(forward_fn.parameters()).device).long()
    attr = captum.attr.GuidedGradCam(forward_fn, forward_fn.embedding_model.conv_block6).attribute(X, target=y)

    if do_norm:
        return attr / attr.max()

    return attr

def gradcam(X, y, forward_fn, do_norm=True):
    """ Computes standard saliency - gradient of the max logit wrt to the input. """
    X = torch.Tensor(X).to(next(forward_fn.parameters()).device)
    y = torch.Tensor(y).to(next(forward_fn.parameters()).device).long()
    attr = captum.attr.LayerGradCam(forward_fn, forward_fn.embedding_model.conv_block6).attribute(X, target=y)
    attr = captum.attr.LayerAttribution.interpolate(attr, (X.shape[-2], X.shape[-1]))

    if do_norm:
        return attr / attr.max()

    return attr
