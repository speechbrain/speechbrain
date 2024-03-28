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
from maskin_maskout import opt_single_mask, interpret_pretrained, all_onesmask
from l2i_eval import l2i_pretrained
import math
from torch.utils.data import Dataset
import torchaudio.transforms as T

import torchaudio.datasets as dts

eps = 1e-10

random.seed(10)


class LJSPEECH_split(dts.LJSPEECH):
    """Create a Dataset for *LJSpeech-1.1* [:footcite:`ljspeech17`].

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"wavs"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(self, root, url, folder_in_archive, download, train=True):
        #super(LJSPEECH_train, self).__init__()
        super().__init__(root, url, folder_in_archive, download)
        #path = os.path.join('LJSpeech-1.1', folder_in_archive)
        #self._flist = glob.glob(path + '/*.wav')
        if train:
            self._flist = self._flist[:10000]
        else:
            self._flist = self._flist[-3000:]
        print('dataset size = ', len(self._flist))

def generate_mixture(s1, s2):
    s1 = s1 / torch.norm(s1)
    s2 = s2 / torch.norm(s2)

    # create the mixture with s2 being the noise (lower gain)
    mix = s1 * 0.8 + (s2 * 0.2)
    mix = mix / mix.max()
    return mix


def fetch_model(url):
    from huggingface_hub import hf_hub_download
    
    REPO_ID = "fpaissan/r"
    
    return hf_hub_download(repo_id=REPO_ID, filename=url)

def generate_overlap(sample, dataset, overlap_multiplier=1, overlap_type='mixtures'):
    #if overlap_type in ['mixtures', 'LJSpeech']:
    pool = [i for i in range(len(dataset))]
    indices = random.sample(pool, overlap_multiplier)
    # print("\n\n Generate overlap called!", indices, " \n\n")

    samples = [
        {k: v for k, v in sample.items()} for _ in range(overlap_multiplier)
    ]
    for i, idx in enumerate(indices):
        if overlap_type == 'mixtures':
            samples[i]["sig"] = generate_mixture(sample["sig"], dataset[idx]["sig"])
        elif overlap_type == 'LJSpeech':
            noise = dataset[idx][0][0]
            tfm = T.Resample(22050, 16000)
            noise = tfm(noise)
            smpl = sample["sig"]

            if noise.shape[0] > smpl.shape[0]:
                noise = noise[:smpl.shape[0]]
            else:
                noise = torch.nn.functional.pad(noise, (0, smpl.shape[0] - noise.shape[0]))
            samples[i]["sig"] = generate_mixture(smpl, noise)
        else:
            smp = sample["sig"] / sample["sig"].pow(2).sum().sqrt()
            noise = torch.randn(sample["sig"].shape)
            noise = noise / noise.pow(2).sum().sqrt()
            samples[i]["sig"] = smp + 0.5*noise
            samples[i]["sig"] = samples[i]["sig"] / samples[i]["sig"].max()
    return samples

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

    root = '/data2/cloned_repos/interpretable_fakereal'
    ljspeech_tr = LJSPEECH_split(root=root,
                           url='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                           folder_in_archive='wavs',
                           download=False, train=True)

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
    #f_emb.load_state_dict(torch.load(fetch_model("embedding_model.ckpt")))
    f_emb.load_state_dict(torch.load(hparams["embedding_model_path"]))

    print(
        "Loading classifier state dict from ", hparams["classifier_model_path"]
    )
    f_cls.load_state_dict(torch.load(hparams["classifier_model_path"]))
    #f_cls.load_state_dict(torch.load(fetch_model("classifier.ckpt")))
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

    d_mosaic = quantus_eval.MosaicDataset(datasets["test"], hparams)
    evaluator = quantus_eval.Evaluator(hparams)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight)
            if hasattr(m, "bias"):
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    if hparams["mrt"]:
        if hparams["mrt_layer"] == -1:
            hparams["classifier"].apply(init_weights)
        if hparams["mrt_layer"] <= 6:
            print("Initialized block 6.")
            hparams["embedding_model"].conv_block6.apply(init_weights)
            if hparams["mrt_layer"] <= 5:
                print("Initialized block 5.")
                hparams["embedding_model"].conv_block5.apply(init_weights)
                if hparams["mrt_layer"] <= 4:
                    print("Initialized block 4.")
                    hparams["embedding_model"].conv_block4.apply(init_weights)
                    if hparams["mrt_layer"] <= 3:
                        print("Initialized block 3.")
                        hparams["embedding_model"].conv_block3.apply(init_weights)
                        if hparams["mrt_layer"] <= 2:
                            print("Initialized block 2.")
                            hparams["embedding_model"].conv_block2.apply(init_weights)
                            if hparams["mrt_layer"] <= 1:
                                print("Initialized block 1.")
                                hparams["embedding_model"].conv_block1.apply(init_weights)

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
        "allones": all_onesmask
    }

    if hparams["exp_method"] == "ao":
        exp_methods["ao"] = interpret_pretrained(psi)

    computed_metrics = [
        "pixel_flip",
        "region_perturbation",
        "max_sensitivity",
        "avg_sensitivity",
        "sparseness",
        "complexity",
        "focus",
        "AI",
        "AD",
        "AG",
        "faithfulness_l2i",
        "inp_fid",
        "accuracy",
        "average"
    ]
    aggregated_metrics = {k: 0.0 for k in computed_metrics}
    samples_interval = hparams["interpret_period"]
    overlap_multiplier = 2

    # cem: this is the stuff I am adding to deal with different noise types
    overlap_type = 'white_noise'
    if overlap_type == 'white_noise':
        dt = datasets["test"]
    elif overlap_type == 'mixtures':
        dt = datasets["test"]
    elif overlap_type == 'LJSpeech':
        dt = ljspeech_tr
    else:
        raise ValueError('Not a valid overlap type')

    discarded = 0
    for idx, base_sample in enumerate(datasets["valid"]):
        if not hparams["add_wham_noise"]:
            overlap_batch = generate_overlap(
                base_sample, dt, overlap_multiplier, overlap_type=overlap_type
            )
            y_batch = torch.Tensor(
                [
                    base_sample["class_string_encoded"]
                    for _ in range(overlap_multiplier)
                ]
            )
            # extract sample
            wavs = [sample["sig"][None] for sample in overlap_batch]
            wavs = torch.stack(wavs).to(run_opts["device"])

        else:
            overlap_batch = base_sample
            y_batch = base_sample["class_string_encoded"]
            
            # extract sample
            wavs = base_sample["sig"].to(run_opts["device"]).unsqueeze(0)

        # preprocess
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        X_oracle, X_stft, X_stft_power = preprocess(wavs, hparams)
        X_stft_phase = spectral_phase(X_stft)

        # assert not hparams["add_wham_noise"], "You should run eval without WHAM! noise."
        if hparams["add_wham_noise"]:
            wavs = combine_batches(
                wavs, iter(hparams["wham_dataset"])
            )
            # wavs = wavs + 0.1*torch.randn(wavs.shape, device=hparams['device'])

            X, X_stft, _ = preprocess(wavs, hparams)
        else:
            X = X_oracle.squeeze(1)

        # X_mosaic, y_mosaic = d_mosaic(X, base_sample["class_string_encoded"])
        X_mosaic, y_mosaic = torch.zeros_like(X), [0 for _ in range(X.shape[0])]

        for o_idx, (X_, X_stft_, X_mosaic_, y_mosaic_, y_batch_) in enumerate(zip(
            X, X_stft, X_mosaic, y_mosaic, y_batch
        )):

            try:
                metrics = evaluator(
                    model_wrap,
                    exp_methods[hparams["exp_method"]],
                    X_[None, None],
                    X_mosaic_[None],  # needed for localization
                    y_mosaic_,
                    y_batch_,
                    X_stft_,
                    hparams["exp_method"],
                    base_sample["id"] + "+" + str(o_idx)
                )

                local = f"Sample={idx+1} "
                local += " ".join(
                    [
                        f"{k}: {v[0] if isinstance(v, list) else v:.3f}"
                        for k, v in metrics.items()
                    ]
                )
                print(local)

                hasNan = False
                for k, v in metrics.items():
                    v = v if not isinstance(v, list) else v[0]
                    if math.isnan(v):
                        hasNan = True

                if hasNan:
                    raise ValueError("Has nan")

                for k, v in metrics.items():
                    aggregated_metrics[k] += v[0] if isinstance(v, list) else v

            except AssertionError as e:
                discarded += 1
                print("Total discarded from quantus are: ", discarded)
                print("Exception was ", str(e))

            except ValueError as e:
                print("Something was NaN here!")
                discarded += 1
                print("Total discarded from quantus are: ", discarded)

        aggregate = f"Aggregated "
        aggregate += " ".join(
            [
                f"{k}: {(v[0] if isinstance(v, list) else v)/((idx+1) * overlap_multiplier):.3f}"
                for k, v in aggregated_metrics.items()
            ]
        )

        # if idx > 20:
            # print("-----------------------------------")
            # print("Breaking loop to transfer data....!!! \n\n")
            # break
            

    for k in aggregated_metrics:
        aggregated_metrics[k] /= len(datasets["valid"]) * overlap_multiplier

    print(aggregated_metrics)

    import json

    os.makedirs("quant_eval", exist_ok=True)
    out_folder = os.path.join(
            hparams["eval_outdir"], f"qualitative_{hparams['experiment_name']}",
            )
    with open(out_folder + "/quant.csv", "w") as f:
        f.write(json.dumps(aggregated_metrics))
