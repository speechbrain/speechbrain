#!/usr/bin/python3
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
from train_piq import InterpreterESC50Brain, tv_loss, dataio_prep_esc50
import pandas as pd
import random
import gradient_based
import quantus_eval
from tqdm import tqdm
from maskin_maskout import opt_single_mask, interpret_pretrained
from l2i_eval import l2i_pretrained

eps = 1e-10

random.seed(10)


def invert_stft_with_phase(hparams):
    def fn(X_int, X_stft_phase):
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
        x_int_sb = hparams["compute_istft"](X_wpsb)
    
        return x_int_sb

    return fn


def fetch_model(url):
    from huggingface_hub import hf_hub_download
    
    REPO_ID = "fpaissan/r"
    
    return hf_hub_download(repo_id=REPO_ID, filename=url)


def preprocess(wavs, hparams):
    """Pre-process wavs."""
    X_stft = hparams["compute_stft"](wavs)
    X_stft_power = sb.processing.features.spectral_magnitude(
        X_stft, power=hparams["spec_mag_power"]
    )

    if not hparams["use_melspectra"]:
        X_stft_logpower = torch.log1p(X_stft_power)
    else:
        X_stft_power = hparams["compute_fbank"](X_stft_power)
        X_stft_logpower = torch.log1p(X_stft_power)

    return X_stft_logpower, X_stft, X_stft_power


if __name__ == "__main__":
    # # This flag enables the inbuilt cudnn auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep_esc50(hparams)
    hparams["label_encoder"] = label_encoder

    # take trained model, make it eval
    f_emb = hparams["embedding_model"].to(run_opts["device"])
    f_cls = hparams["classifier"].to(run_opts["device"])
    print(
        "Loading embedding_model state dict from ",
        hparams["embedding_model_path"],
    )
    f_emb.load_state_dict(torch.load(fetch_model("embedding_model.ckpt")))
    print(
        "Loading classifier state dict from ", hparams["classifier_model_path"]
    )
    f_cls.load_state_dict(torch.load(fetch_model("classifier.ckpt")))
    f_emb.eval()
    f_cls.eval()

    # freeze weights
    f_emb.requires_grad_(False)
    f_cls.requires_grad_(False)

    if hparams["exp_method"] == "ao" or hparams["exp_method"] == "l2i":
        psi = hparams["psi_model"].to(run_opts["device"])
        print("Loading PSI state dict from ", hparams["pretrained_PIQ"])
        psi.load_state_dict(torch.load(hparams["pretrained_PIQ"]))
        psi.eval()
        hparams["psi_model"] = psi

    hparams["embedding_model"] = f_emb
    hparams["classifier"] = f_cls

    if hparams["exp_method"] == "l2i":
        # load theta as well...
        hparams["nmf_decoder"].load_state_dict(
            torch.load(hparams["nmf_decoder_path"])
        )
        hparams["nmf_decoder"].to(run_opts["device"])
        hparams["psi"] = psi

    model_wrap = quantus_eval.Model(
        hparams,
        f_emb,
        f_cls,
        repr_="ao" == hparams["exp_method"] or "l2i" in hparams["exp_method"],
    )
    model_wrap.eval()

    exp_methods = {
        "saliency": gradient_based.saliency,
        "IG": gradient_based.ig,
        "smoothgrad": gradient_based.smoothgrad,
        "guided_backprop": gradient_based.guided_backprop,
        "guided_gradcam": gradient_based.guided_gradcam,
        "gradcam": gradient_based.gradcam,
        "shap": gradient_based.shap,
        "guided_IG": gradient_based.guided_IG,
        "single_maskinout": opt_single_mask,
        "l2i": l2i_pretrained(hparams, run_opts),
    }

    if hparams["exp_method"] == "ao":
        exp_methods["ao"] = interpret_pretrained(psi)

    assert len(sys.argv) > 2, "Did you pass the sample path?"
    wavs, sr = torchaudio.load(hparams["sample"])
    wavs = torchaudio.transforms.Resample(sr, hparams["sample_rate"])(wavs).to(run_opts["device"]) # make it 16kHz

    # preprocess
    if wavs.ndim == 3: wavs = wavs.squeeze(1)
    X_oracle, X_stft, X_stft_power = preprocess(wavs, hparams)
    X_stft_phase = spectral_phase(X_stft)

    X = X_oracle.squeeze(1)
    # X = torch.log1p(X)

    if hparams["exp_method"] == "ao":
        int_ = interpret_pretrained(hparams["psi_model"])(X.unsqueeze(1), None, model_wrap)
    elif hparams["exp_method"] == "l2i":
        int_ = l2i_pretrained(hparams, run_opts)(X.unsqueeze(1), None, model_wrap)

    X = X[:, :int_.shape[2], :]
    pred = model_wrap(X)[0].argmax(1).item()
    print("predicted class is ", pred)
    X_stft_phase = X_stft_phase[:, :int_.shape[2], :]
    int_ = int_[:, :, :X.shape[-2], :]

    with torch.no_grad():
        plt.imshow(int_.squeeze().t().cpu(), origin="lower")
        plt.savefig("int.png")

    gradient_based.save_waves((X, X_stft_phase), int_, [hparams["exp_method"]], invert_stft_with_phase(hparams))

