""" Evaluation metrics for interpretability

Authors
    * Francesco Paissan 2023, 2024
    * Cem Subakan 2023, 2024
"""

import copy
import os

import numpy as np
import quantus
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantus.helpers.utils import expand_indices

eps = 1e-9


class QuantusEvalWrapper(nn.Module):
    def __init__(
        self,
        use_stft2mel,
        use_melspectra,
        compute_stft,
        compute_fbank,
        repr_=False,
    ):
        super().__init__()
        self.returnrepr = repr_
        self.use_stft2mel = use_stft2mel
        self.use_melspectra = use_melspectra
        self.compute_fbank = compute_fbank
        self.compute_stft = compute_stft

    def forward(self, x, embedding_model, classifier):
        print(x.shape)
        breakpoint()
        self.cnn14 = False
        if str(self.embedding_model.__class__.__name__) == "Cnn14":
            self.cnn14 = True

        x = x.float()
        x = torch.log1p(self.compute_stft(x))
        if self.use_stft2mel and not self.use_melspectra:
            x = self.compute_fbank(x.squeeze(1)).unsqueeze(1)
            x = torch.log1p(x)

        if x.ndim == 4:
            x = x.squeeze(1)

        temp = embedding_model(x)
        if isinstance(temp, tuple):
            embeddings, f_I = temp
        else:
            embeddings, f_I = temp, temp

        if embeddings.ndim == 4:
            embeddings = embeddings.mean((-1, -2))

        predictions = classifier(embeddings).squeeze(1)

        if self.returnrepr:
            return predictions, f_I

        return predictions


def wrap_gradient_based(explain_fn, forw=True):
    def fn(model, inputs, targets, **kwargs):
        inputs = torch.Tensor(inputs).to(next(model.parameters()).device)
        ex = explain_fn(inputs, targets, model).detach().cpu().numpy()
        return ex

    return fn


@torch.no_grad()
def compute_fidelity(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
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


@torch.no_grad()
def truncated_gaussian_noise(
    arr,
    indices,
    indexed_axes,
    perturb_mean=0,
    perturb_std=0.15,
    **kwargs,
):
    """
    Add gaussian noise to the input at indices.

    Parameters
    ----------
    arr: np.ndarray
         Array to be perturbed.
    indices: int, sequence, tuple
        Array-like, with a subset shape of arr.
    indexed_axes: sequence
        The dimensions of arr that are indexed.
        These need to be consecutive, and either include the first or last dimension of array.
    perturb_mean (float):
        The mean for gaussian noise.
    perturb_std (float):
        The standard deviation for gaussian noise.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    arr_perturbed: np.ndarray
         The array which some of its indices have been perturbed.
    """

    indices = expand_indices(arr, indices, indexed_axes)
    noise = np.abs(
        np.random.normal(
            loc=0,
            scale=(np.max(arr) - np.min(arr)) * perturb_std,
            size=arr.shape,
        )
    )

    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = (arr_perturbed + noise)[indices]

    return arr_perturbed


class Evaluator:
    def __init__(self, hparams, X_shape=(1, 431, 513)):
        self.hparams = hparams

        self.max_sensitivity = quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=truncated_gaussian_noise,
            similarity_func=quantus.similarity_func.difference,
            return_aggregate=True,
            abs=True,
        )

        self.avg_sensitivity = quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=truncated_gaussian_noise,
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

    def get_faith_metrics(self, X, model, method, explain_fn):
        model = model.eval()
        metrics = {}

        predictions = model(X)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = predictions.softmax(1)
        y = predictions.argmax(1)

        if "mask" not in method and "ao" != method:
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
        # changing to model(maskout)
        maskout_preds = model(maskout)
        if isinstance(maskout_preds, tuple):
            maskout_preds = maskout_preds[0]
        maskout_preds = maskout_preds.softmax(1)

        metrics["AI"] = compute_AI(maskin_preds, predictions).item()
        metrics["AD"] = compute_AD(maskin_preds, predictions).item()
        metrics["AG"] = compute_AG(maskin_preds, predictions).item()
        metrics["faithfulness_l2i"] = compute_faithfulness(
            predictions, maskout_preds
        ).item()
        metrics["inp_fid"] = compute_fidelity(maskin_preds, predictions).item()

        return metrics, inter, y

    def __call__(
        self,
        model,
        X,
        X_mosaic,
        y_mosaic,
        y,
        X_stft,
        method,
        id_,
        device="cuda",
    ):
        """computes quantus metrics sample-wise"""
        if model.training:
            model.eval()

        explain_fn = None  # TODO: modify

        out_folder = os.path.join(
            self.hparams["eval_outdir"],
            f"qualitative_{self.hparams['experiment_name']}",
            id_,
        )
        os.makedirs(
            os.path.join(
                self.hparams["eval_outdir"],
                f"qualitative_{self.hparams['experiment_name']}",
            ),
            exist_ok=True,
        )
        os.makedirs(out_folder, exist_ok=True)

        metrics, inter, y_pred = self.get_faith_metrics(
            X, model, method, explain_fn
        )

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

        metrics["average"] = inter.mean().item()
        if method != "allones":
            metrics["sparseness"] = self.sparseness(**quantus_inp)
            metrics["complexity"] = self.complexity(**quantus_inp)
        metrics["accuracy"] = (y.item() == y_pred.item()) * 100

        return metrics
