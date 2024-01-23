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
import gradient_based
import quantus
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

eps = 1e-9


class Model(nn.Module):
    def __init__(self, hparams, embedding_model, classifier, repr_=False):
        super().__init__()
        self.returnrepr = repr_
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.hparams = hparams

        self.cnn14 = False
        if str(self.embedding_model.__class__.__name__) == "Cnn14":
            self.cnn14 = True

    def forward(self, x):
        x = x.float()
        if self.hparams["use_stft2mel"] and not self.hparams["use_melspectra"]:
            x = torch.expm1(x)
            x = self.hparams["compute_fbank"](x.squeeze(1))[None]
            x = torch.log1p(x)

        if x.ndim == 4:
            x = x.squeeze(1)

        temp = self.embedding_model(x)
        if isinstance(temp, tuple):
            embeddings, f_I = temp
        else:
            embeddings, f_I = temp, temp

        if embeddings.ndim == 4:
            embeddings = embeddings.mean((-1, -2))

        predictions = self.classifier(embeddings).squeeze(1)

        if self.returnrepr:
            return predictions, f_I

        return predictions


def wrap_gradient_based(explain_fn, forw=True):
    def fn(model, inputs, targets, **kwargs):
        inputs = torch.Tensor(inputs).to(next(model.parameters()).device)
        ex = explain_fn(inputs, targets, model).cpu().numpy()
        return ex

    return fn


class MosaicDataset:
    """Generate mosaics to compute Focus!"""

    def __init__(self, dataset_test, hparams):
        self.hparams = hparams
        self.dataset = dataset_test  # used to sample

    def generate_mosaic(self, elements):
        labels = [1, 1, 0, 0]

        # apply same randomization to two lists
        temp = list(zip(elements, labels))
        random.shuffle(temp)
        elements, labels = zip(*temp)
        elements, labels = list(elements), list(labels)

        mosaic = torch.zeros(elements[0].shape[1] * 2, elements[0].shape[2] * 2)

        mosaic[0:417, 0:513] = elements[0]
        mosaic[0:417, 513:] = elements[1]
        mosaic[417:, 0:513] = elements[2]
        mosaic[417:, 513:] = elements[3]

        return mosaic[None], labels

    def sample_wlabel(self, label):
        remove_from_pool = []
        not_ok_labels = [label]
        pool = [i for i in range(len(self.dataset))]
        negative_idx = []
        for wanted in [True, False, False]:
            for el in remove_from_pool:
                pool.remove(el)
                remove_from_pool.remove(el)

            if wanted:
                s_label = label + 1  # not equal
                while s_label != label:
                    s = np.random.choice(pool)
                    s_label = self.dataset[s]["class_string_encoded"]

                m21_idx = s  # second positive sample
                remove_from_pool.append(s)

            else:
                s_label = label
                while s_label in not_ok_labels:
                    s = np.random.choice(pool)
                    s_label = self.dataset[s]["class_string_encoded"]

                negative_idx.append(s)  # append negative examples
                remove_from_pool.append(s)

        return m21_idx, negative_idx[0], negative_idx[1]

    def preprocess(self, sample):
        """Pre-process wavs."""
        wavs = sample["sig"][None]
        X_stft = self.hparams["compute_stft"](wavs)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams["spec_mag_power"]
        )

        if not self.hparams["use_melspectra"]:
            X_stft_logpower = torch.log1p(X_stft_power)
        else:
            X_stft_power = self.hparams["compute_fbank"](X_stft_power)
            X_stft_logpower = torch.log1p(X_stft_power)

        return X_stft_logpower[:, :425, :]

    def __call__(self, batch, c_m):
        mosaics = []
        mosaic_labels = []
        for sample in batch:
            # returns sample with same label, and two different labels
            idx_cm, idx_n1, idx_n2 = self.sample_wlabel(c_m)

            m12 = self.preprocess(self.dataset[idx_cm])
            m21 = self.preprocess(self.dataset[idx_n1])
            m22 = self.preprocess(self.dataset[idx_n2])

            temp = self.generate_mosaic(
                [sample[None][:, :417, :], m12, m21, m22]
            )

            mosaics.append(temp[0])
            mosaic_labels.append(torch.Tensor(temp[1]))

        mosaics = torch.stack(mosaics)
        mosaic_labels = torch.stack(mosaic_labels)

        mosaics = mosaics[:, :, :833, :1025]

        return mosaics, mosaic_labels


@torch.no_grad()
def compute_fidelity(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    # predictions = F.softmax(predictions, dim=1)
    # theta_out = F.softmax(theta_out, dim=1)

    pred_cl = torch.argmax(predictions, dim=1)
    k_top = torch.topk(theta_out, k=1, dim=1)[1]

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)

    return temp


@torch.no_grad()
def compute_faithfulness(predictions, predictions_masked):
    # get the prediction indices
    pred_cl = predictions.argmax(dim=1, keepdim=True)

    # get the corresponding output probabilities
    predictions_selected = torch.gather(predictions, dim=1, index=pred_cl)
    predictions_masked_selected = torch.gather(
        predictions_masked, dim=1, index=pred_cl
    )

    faithfulness = (
        predictions_selected - predictions_masked_selected
    ).squeeze()

    return faithfulness


@torch.no_grad()
def compute_AD(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    predictions = F.softmax(predictions, dim=1)
    theta_out = F.softmax(theta_out, dim=1)

    pc = torch.gather(
        predictions, dim=1, index=predictions.argmax(1, keepdim=True)
    ).squeeze()
    oc = torch.gather(
        theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
    ).squeeze()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (F.relu(pc - oc) / (pc + eps)) * 100

    return temp


@torch.no_grad()
def compute_AI(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    # predictions = F.softmax(predictions, dim=1)
    # theta_out = F.softmax(theta_out, dim=1)

    pc = torch.gather(
        predictions, dim=1, index=predictions.argmax(1, keepdim=True)
    ).squeeze()
    oc = torch.gather(
        theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
    ).squeeze()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (pc < oc).float() * 100

    return temp


@torch.no_grad()
def compute_AG(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    # predictions = F.softmax(predictions, dim=1)
    # theta_out = F.softmax(theta_out, dim=1)

    pc = torch.gather(
        predictions, dim=1, index=predictions.argmax(1, keepdim=True)
    ).squeeze()
    oc = torch.gather(
        theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
    ).squeeze()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (F.relu(oc - pc) / (1 - pc + eps)) * 100

    return temp


class Evaluator:
    def __init__(self, hparams, X_shape=(1, 431, 513)):
        self.hparams = hparams
        self.first = True

        self.pixel_flipping = quantus.PixelFlipping(
            features_in_step=X_shape[2],
            perturb_baseline="black",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            return_auc_per_sample=True,
            abs=True,
        )

        self.region_perturb = quantus.RegionPerturbation(
            patch_size=14,
            regions_evaluation=10,
            perturb_baseline="uniform",
            return_aggregate=True,
            normalise=True,
            abs=True,
        )

        self.max_sensitivity = quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            return_aggregate=True,
            abs=True,
        )

        self.avg_sensitivity = quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            return_aggregate=True,
            abs=True,
        )

        self.sparseness = quantus.Sparseness(return_aggregate=True, abs=True)

        self.complexity = quantus.Complexity(return_aggregate=True, abs=True)

        self.focus = quantus.Focus(
            mosaic_shape=[1, X_shape[1] * 2, X_shape[2] * 2],
            return_aggregate=True,
            abs=True,
        )

    def compute_ours(self, X, model, method, explain_fn):
        metrics = {}

        predictions = model(X)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.softmax(1)
        y = predictions.argmax(1)

        if not "mask" in method and "ao" != method:
            inter = explain_fn(X, y, model)
        else:
            inter = explain_fn(X, y, model)

        if method == "ao" or method == "l2i":
            X = X[:, :, : inter.shape[2], : inter.shape[3]]

        maskin = X * inter
        maskin_preds = model(maskin)
        if isinstance(maskin_preds, tuple):
            maskin_preds = maskin_preds[0]
        maskin_preds = maskin_preds.softmax(1)

        maskout = X * (1 - inter)
        maskout_preds = model(maskin)
        if isinstance(maskout_preds, tuple):
            maskout_preds = maskout_preds[0]
        maskout_preds = maskout_preds.softmax(1)

        if self.first:
            gradient_based.save_ints(
                [X.squeeze()],
                [inter.squeeze()],
                [method],
                fname=f"{method}.png",
            )

        metrics["AI"] = compute_AI(maskin_preds, predictions).item()
        metrics["AD"] = compute_AD(maskin_preds, predictions).item()
        metrics["AG"] = compute_AG(maskin_preds, predictions).item()
        metrics["faithfulness_l2i"] = compute_faithfulness(
            predictions, maskout_preds
        ).item()
        metrics["inp_fid"] = compute_fidelity(maskin_preds, predictions).item()

        return metrics, inter

    def __call__(
        self, model, explain_fn, X, X_mosaic, y_mosaic, y, X_stft, method, id_, device="cuda"
    ):
        """computes quantus metrics sample-wise"""
        if model.training:
            model.eval()

        if self.first:
            out_folder = os.path.join(
                    f"qualitative_{method}", id_
                    )
            os.makedirs(f"qualitative_{method}", exist_ok=True)
            os.makedirs(out_folder, exist_ok=True)

        metrics, inter = self.compute_ours(X, model, method, explain_fn)

        self.debug_files(X_stft, X, inter, id_, out_folder, )

        X = X.clone().detach().cpu().numpy()
        y = y.clone().detach().cpu().numpy()
        attr = inter.clone().detach().cpu().numpy()
        y_mosaic = np.array(y_mosaic)[None]
        wrap_explain_fn = wrap_gradient_based(
            explain_fn, forw="maskin" in method
        )

        # matches interpretations shapes
        X = X[:, :, : attr.shape[2], : attr.shape[3]]

        quantus_inp = {
            "model": model,
            "x_batch": X,  # quantus expects the batch dim
            "y_batch": np.array([y], dtype=int),
            "a_batch": attr,
            "softmax": False,
            "device": device,
            "explain_func": wrap_explain_fn,
        }

        metrics["max_sensitivity"] = self.max_sensitivity(**quantus_inp)

        metrics["avg_sensitivity"] = self.avg_sensitivity(**quantus_inp)

        metrics["sparseness"] = self.sparseness(**quantus_inp)

        metrics["complexity"] = self.complexity(**quantus_inp)

        if method != "l2i" and False:
            quantus_inp[
                "x_batch"
            ] = X_mosaic  # quantus expects the batch dim_mosaic
            quantus_inp["a_batch"] = None
            metrics["focus"] = self.focus(
                custom_batch=y_mosaic,  # look here https://github.com/understandable-machine-intelligence-lab/Quantus/blob/c32da2b6e39f41b50572d1e4a4ddfc061e0bb8b2/quantus/metrics/localisation/focus.py#L307
                **quantus_inp,
            )

        if self.first:
            self.first = not self.first

        return metrics

    def debug_files(self, X_stft, X_logpower, interpretation, fname="test", out_folder="."):
        """The helper function to create debugging images"""
        X_stft_phase = spectral_phase(X_stft[None])

        X = torch.expm1(X_logpower)[0, ..., None]
        x_inp = self.invert_stft_with_phase(X, X_stft_phase)

        torchaudio.save(
                f"{os.path.join(out_folder, fname)}_original.wav",
                x_inp.cpu(),
                sample_rate=16000
                )

        int_ = torch.expm1(X_logpower[0, ..., None] * interpretation[0, ..., None])
        x_int = self.invert_stft_with_phase(int_, X_stft_phase)

        torchaudio.save(
                f"{os.path.join(out_folder, fname)}_int.wav",
                x_int.cpu(),
                sample_rate=16000
                )
        plt.figure(figsize=(11, 10), dpi=100)

        plt.subplot(311)
        torch.save(X_logpower, os.path.join(out_folder, "x_logpower.pt"))
        plt.imshow(X_logpower.squeeze().t().cpu(), origin="lower")
        plt.title("input")
        plt.colorbar()

        plt.subplot(312)
        torch.save(interpretation, os.path.join(out_folder, "interpretation.pt"))
        X_masked = interpretation.squeeze().t().cpu()
        plt.imshow(X_masked.data.cpu(), origin="lower")
        plt.colorbar()
        plt.title("mask")

        plt.subplot(313)
        plt.imshow((X_logpower.squeeze().t().cpu() * X_masked).cpu(), origin="lower")
        plt.colorbar()
        plt.title("masked")

        out_fname_plot = os.path.join(
            out_folder,
            f"{fname}.png"
        )

        plt.savefig(out_fname_plot)
        plt.close()

    def invert_stft_with_phase(self, X_int, X_stft_phase):
        """Inverts STFT spectra given phase."""
        X_stft_phase_sb = torch.cat(
            (
                torch.cos(X_stft_phase).unsqueeze(-1),
                torch.sin(X_stft_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

        X_stft_phase_sb = X_stft_phase_sb[:, : X_int.shape[1], :, :]
        if X_int.ndim == 3:
            X_int = X_int.unsqueeze(-1)
        X_wpsb = X_int * X_stft_phase_sb
        x_int_sb = self.hparams["compute_istft"](X_wpsb)

        return x_int_sb
