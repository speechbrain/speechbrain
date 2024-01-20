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

def saliency(X, forward_fn, do_norm=True):
    """ Computes standard saliency - gradient of the max logit wrt to the input. """
    X.requires_grad = True

    predictions = forward_fn(X)

    score, indices = torch.max(predictions, 1)

    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()

    # get max along channel axis
    slc, _ = torch.max(torch.abs(X.grad), dim=0)

    if do_norm:
        # normalize to [0..1] -- minmax norm
        slc = (slc - slc.min())/(slc.max()-slc.min())

    return slc

def smoothgrad(X, forward_fn, steps=50, noise_level=0.15, guidance=False):
    """ Runs smoothgrad - gauss noise on input before saliency map. """
    # derive sigma from noise level -- see Fig 3. https://arxiv.org/pdf/1706.03825.pdf
    sigma = noise_level * (X.max() - X.min())

    maps = []
    for _ in range(steps):
        X_perturb = torch.randn_like(X) * sigma + X
        slc = saliency(X_perturb.detach(), forward_fn)

        maps.append(slc)
    
    return torch.stack(maps).mean(0)

def ig(X, forward_fn, steps=200):
    """ Computes IG starting from X_baseline to X. """
    X_baseline = torch.zeros_like(X)
    X_v = X - X_baseline
    maps = []
    for t in range(steps):
        X_t = X_baseline + t / (steps - 1) * X_v   # q + x m
        slc = saliency(X_t.detach(), forward_fn, do_norm=False)

        maps.append(slc)

    # accumulate the gradients over the path -- eq. 3 in paper https://arxiv.org/pdf/1703.01365.pdf
    ig = torch.stack(maps).mean(0) * X_v[0]

    # normalize to [0..1] -- minmax norm
    ig = (ig - ig.min()) / (ig.max() - ig.min())

    return ig

def saliency_guided(X, X_oracle, forward_fn, do_norm=True):
    """ Computes guided saliency - the guidance is supposed to be != X, and 
    is used as concept-activating direction.
    We compute 'how much would this logit change going towards this sample?'
    """
    assert (X - X_oracle).sum() != 0, "X matches oracle."
    slc = saliency(X, forward_fn, do_norm=False)

    # compute the direction and normalize
    g_v = (X_oracle - X)[0]
    g_v /= torch.norm(g_v)

    # directional gradient -- how is this logit affected from a change towards the oracle?
    slc = slc * g_v
    print(slc.shape, g_v.shape)
    breakpoint()
    exit()

    if do_norm:
        # normalize to [0..1] -- minmax norm
        slc = (slc - slc.min())/(slc.max()-slc.min())

    return slc

def smoothgrad_guided(X, X_oracle, forward_fn, steps=50, noise_level=0.15):
    """ Runs smoothgrad - gauss noise on input before guided saliency map. """
    # derive sigma from noise level -- see Fig 3. https://arxiv.org/pdf/1706.03825.pdf
    sigma = noise_level * (X.max() - X.min())

    maps = []
    for _ in range(steps):
        X_perturb = torch.randn_like(X) * sigma + X
        slc = saliency_guided(X_perturb.detach(), X_oracle, forward_fn)

        maps.append(slc)
    
    return torch.stack(maps).mean(0)
