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
from train_piq import InterpreterESC50Brain, tv_loss, dataio_prep
import pandas as pd
import random
import gradient_based
import quantus_eval
from tqdm import tqdm
from maskin_maskout import opt_single_mask

eps = 1e-10


def generate_mixture(s1, s2):
    s1 = s1 / torch.norm(s1)
    s2 = s2 / torch.norm(s2)

    # create the mixture with s2 being the noise (lower gain)
    mix = (s1 * 0.8 + (s2 * 0.2))
    mix = mix / mix.max()

    return mix


def generate_overlap(sample, dataset, overlap_multiplier=1):
    pool = [i for i in range(len(dataset))]
    indices = random.sample(pool, overlap_multiplier)

    samples = [{k: v for k, v in sample.items()} for _ in range(overlap_multiplier)]
    for i, idx in enumerate(indices):
        samples[i]["sig"] = generate_mixture(sample["sig"], dataset[idx]["sig"])

    return samples


def interpret_pretrained(interpreter):
    interpreter = interpreter.eval()

    @torch.no_grad()
    def interpret_ao(ft, model):
        predictions, fI = model(ft)
        pred_cl = predictions.argmax(1)

        temp = interpreter(fI, pred_cl)[0]
        temp = temp[:, :, :ft.shape[2], :ft.shape[3]]

        plt.imshow(ft.squeeze().t().cpu())
        plt.savefig("inp.png")
        plt.imshow(temp.squeeze().t().cpu())
        plt.savefig("int.png")
        breakpoint()

        return torch.sigmoid(temp)

    return interpret_ao


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

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

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
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    brain = InterpreterESC50Brain(
        modules=hparams["modules"], hparams=hparams, run_opts=run_opts
    )

    # take trained model, make it eval
    f_emb = hparams["embedding_model"].to(brain.device)
    f_cls = hparams["classifier"].to(brain.device)
    f_emb.eval()
    f_cls.eval()

    brain.hparams.compute_stft.to(brain.device)
    brain.hparams.compute_fbank.to(brain.device)

    # freeze weights
    f_emb.requires_grad_(False)
    f_cls.requires_grad_(False)
    if hparams["exp_method"] == "ao":
        if hparams["pretrained_PIQ"] is None:
            raise AssertionError(
                "You should specificy pretrained model for finetuning."
            )

    psi = hparams["psi_model"]
    if hparams["pretrained_PIQ"]:
        print("Loading pretrained PIQ from ", hparams["pretrained_PIQ"])
        # run_on_main(hparams["load_pretrained"].collect_files)
        # hparams["load_pretrained"].load_collected()
        psi.load_state_dict(
                torch.load(hparams["pretrained_PIQ"])
                )

    d_mosaic = quantus_eval.MosaicDataset(datasets["test"], brain)
    evaluator = quantus_eval.Evaluator()

    model_wrap = quantus_eval.Model(
        brain.hparams.embedding_model, brain.hparams.classifier, repr_="ao" == hparams["exp_method"]
    )
    model_wrap.eval()

    exp_methods = {
            "saliency": gradient_based.saliency,
            "IG": gradient_based.ig,
            "smoothgrad": gradient_based.smoothgrad,
            "single_maskinout": opt_single_mask,
            "ao": interpret_pretrained(psi),
            }

    computed_metrics = [
            "pixel_flip", "region_perturbation", "max_sensitivity",
            "avg_sensitivity", "sparseness", "complexity", "focus",
            "AI", "AD", "AG", "faithfulness_l2i", "inp_fid"
            ]
    aggregated_metrics = {k: 0. for k in computed_metrics}
    samples_interval = hparams["interpret_period"]
    overlap_multiplier = 2
    for idx, base_sample in tqdm(enumerate(datasets["valid"]), desc="Running eval..."):
        overlap_batch = generate_overlap(base_sample, datasets["test"], overlap_multiplier)
        y_batch = torch.Tensor(
                [base_sample["class_string_encoded"] for _ in range(overlap_multiplier)]
                )

        # extract sample
        wavs = [sample["sig"][None] for sample in overlap_batch]
        wavs = torch.stack(wavs).to(brain.device)

        # preprocess
        X_oracle, X_stft, X_stft_power = brain.preprocess(wavs.squeeze(1))
        X_stft_phase = spectral_phase(X_stft)

        # assert not hparams["use_wham"], "You should run eval on overlap test."
        if hparams["add_wham_noise"]:
            wavs = combine_batches(wavs.squeeze(1), iter(hparams["wham_dataset"]))

            X, _, _ = brain.preprocess(wavs)

        else:
            X = X_oracle.squeeze(1)

        X_mosaic, y_mosaic = d_mosaic(X, base_sample["class_string_encoded"])

        for X_, X_mosaic_, y_mosaic_, y_batch_ in zip(
                X, X_mosaic, y_mosaic, y_batch
                ):
            metrics = evaluator(
                model_wrap,
                exp_methods[hparams["exp_method"]],
                X_[None, None],
                X_mosaic_[None],  # needed for localization
                y_mosaic_,
                y_batch_,
                hparams["exp_method"]
            )

            for k, v in metrics.items():
                aggregated_metrics[k] += v[0] if isinstance(v, list) else v

    for k in aggregated_metrics:
        aggregated_metrics[k] /= len(datasets["valid"]) * overlap_multiplier

    print(aggregated_metrics)
    os.makedirs("quant_eval", exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv("quant_eval/" + hparams["exp_method"] + ".csv")

