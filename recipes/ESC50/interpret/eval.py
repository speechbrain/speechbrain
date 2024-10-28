#!/usr/bin/python3
"""This file performs out-of-distribution (OOD) evaluation of interpreters.

To run this recipe, use the following command:
    python eval.py hparams/<config>.yaml --data_folder /yourpath/esc50 --overlap_type <mixtures/ljspeech/white_noise> --add_wham_noise False
    Please refer to README.md for more details.

Authors
    * Francesco Paissan 2024
    * Cem Subakan 2024
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.datasets as dts
import torchaudio.transforms as T
from esc50_prepare import dataio_prep, prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from train_l2i import L2I
from train_lmac import LMAC
from wham_prepare import prepare_wham

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

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
        # super(LJSPEECH_train, self).__init__()
        super().__init__(root, url, folder_in_archive, download)
        # path = os.path.join('LJSpeech-1.1', folder_in_archive)
        # self._flist = glob.glob(path + '/*.wav')
        if train:
            self._flist = self._flist[:10000]
        else:
            self._flist = self._flist[-3000:]
        print("dataset size = ", len(self._flist))


class ESCContaminated(torch.utils.data.Dataset):
    """ESC50 Contaminated dataset

    Arguments
    ---------
    esc50_ds : dataset
        the ESC50 dataset as per training.
    cont_d : dataset
        the contamination dataset.
    overlap_multiplier : int
        number of overlaps
    overlap_type : str
        one of "mixtures" or "LJSpeech" or "white_noise"
    """

    def __init__(
        self, esc50_ds, cont_d, overlap_multiplier=2, overlap_type="mixtures"
    ):
        super().__init__()

        self.esc50_ds = esc50_ds
        self.cont_d = cont_d
        self.overlap_multiplier = overlap_multiplier
        self.overlap_type = overlap_type

    def generate_mixture(self, s1, s2):
        s1 = s1 / torch.norm(s1)
        s2 = s2 / torch.norm(s2)

        # create the mixture with s2 being the noise (lower gain)
        mix = s1 * 0.8 + (s2 * 0.2)
        mix = mix / mix.max()
        return mix

    def __len__(self):
        return len(self.esc50_ds)

    def __getitem__(
        self,
        idx_mix: int,
    ):
        sample = self.esc50_ds[idx_mix]

        pool = [i for i in range(len(self.cont_d))]
        indices = random.sample(pool, self.overlap_multiplier)

        samples = [
            {k: v for k, v in sample.items()}
            for _ in range(self.overlap_multiplier)
        ]
        for i, idx in enumerate(indices):
            if self.overlap_type == "mixtures":
                samples[i]["sig"] = self.generate_mixture(
                    sample["sig"], self.cont_d[idx]["sig"]
                )

            elif self.overlap_type == "LJSpeech":
                noise = self.cont_d[idx][0][0]
                tfm = T.Resample(22050, 16000)
                noise = tfm(noise)
                smpl = sample["sig"]

                if noise.shape[0] > smpl.shape[0]:
                    noise = noise[: smpl.shape[0]]
                else:
                    noise = torch.nn.functional.pad(
                        noise, (0, smpl.shape[0] - noise.shape[0])
                    )
                samples[i]["sig"] = self.generate_mixture(smpl, noise)

            elif self.overlap_type == "white_noise":
                smp = sample["sig"] / sample["sig"].pow(2).sum().sqrt()
                noise = torch.randn(sample["sig"].shape)
                noise = noise / noise.pow(2).sum().sqrt()
                samples[i]["sig"] = smp + 0.5 * noise
                samples[i]["sig"] = samples[i]["sig"] / samples[i]["sig"].max()

            else:
                raise ValueError("Overlap type not implemented.")

        return sb.dataio.batch.PaddedBatch(samples)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if hparams["add_wham_noise"]:
        print(
            "CAREFUL! You are running ID evaluation. If you want to run OOD, use add_wham_noise=False."
        )
    ljspeech_tr = None
    if hparams["ljspeech_path"] is not None:
        os.makedirs(hparams["ljspeech_path"], exist_ok=True)
        ljspeech_tr = LJSPEECH_split(
            root=hparams["ljspeech_path"],
            url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
            folder_in_archive="wavs",
            download=True,
            train=True,
        )

    if hparams["overlap_type"] == "LJSpeech":
        assert (
            ljspeech_tr is not None
        ), "Specify a path if you want to generate OOD with LJSpeech."

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

    # create WHAM dataset according to hparams
    if "wham_folder" in hparams:
        hparams["wham_dataset"] = prepare_wham(
            hparams["wham_folder"],
            hparams["add_wham_noise"],
            hparams["sample_rate"],
            hparams["signal_length_s"],
            hparams["wham_audio_folder"],
        )
        assert hparams["signal_length_s"] == 5, "Fix wham sig length!"
        assert hparams["out_n_neurons"] == 50, "Fix number of outputs classes!"

    assert hparams["use_pretrained"], "Load a model checkpoint during eval."
    if "pretrained_esc50" in hparams and hparams["use_pretrained"]:
        print("Loading model...")
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    hparams["embedding_model"].to(run_opts["device"])
    hparams["classifier"].to(run_opts["device"])
    hparams["embedding_model"].eval()
    hparams["classifier"].eval()

    overlap_type = hparams["overlap_type"]
    if overlap_type == "white_noise":
        overlap_dataset = datasets["test"]
    elif overlap_type == "mixtures":
        overlap_dataset = datasets["test"]
    elif overlap_type == "LJSpeech":
        overlap_dataset = ljspeech_tr
    else:
        raise ValueError("Not a valid overlap type")

    ood_dataset = ESCContaminated(
        datasets["valid"], overlap_dataset, overlap_type=overlap_type
    )

    # if add_wham_noise == True then ood_dataset is simply in domain
    if hparams["add_wham_noise"]:
        ood_dataset = datasets["valid"]

    assert (
        hparams["pretrained_interpreter"] is not None
    ), "You need to specify a path for the pretrained_interpreter!"
    hparams["psi_model"].load_state_dict(
        torch.load(hparams["pretrained_interpreter"], map_location="cpu")
    )

    if hparams["int_method"] == "lmac":
        Interpreter = LMAC(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
        )
    elif hparams["int_method"] == "l2i":
        # hparams["nmf_decoder"].load_state_dict(
        #     torch.load(hparams["nmf_decoder_path"], map_location="cpu")
        # )
        hparams["nmf_decoder"].to(run_opts["device"])
        hparams["nmf_decoder"].eval()

        Interpreter = L2I(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
        )

    if hparams["single_sample"] is None:
        Interpreter.evaluate(
            test_set=ood_dataset,
            min_key="loss",
            progressbar=True,
            test_loader_kwargs=(
                {"collate_fn": lambda x: x[0], "batch_size": 1}
                if not hparams["add_wham_noise"]
                else {"batch_size": 2}
            ),
        )

    else:
        wav, sr = torchaudio.load(hparams["single_sample"])
        wav = T.Resample(sr, hparams["sample_rate"])(wav).to(run_opts["device"])

        with torch.no_grad():
            X_int, _, X_stft_phase, X_orig = (
                Interpreter.interpret_computation_steps(wav)
            )

        # make sure shapes are ok
        X_int = X_int.transpose(1, 2)
        X_orig = X_orig[:, : X_int.shape[1]]
        X_stft_phase = X_stft_phase[:, : X_int.shape[1]]

        def plot_spec(X, suffix=""):
            X = X.expm1()
            X = X ** (1 / 3)

            plt.figure(figsize=(5, 5))
            plt.matshow(
                X.cpu().numpy()[0].T,
                aspect="auto",
                origin="lower",
                cmap="inferno",
            )
            plt.axis("off")
            plt.savefig(
                ".".join(hparams["single_sample"].split(".")[:-1])
                + f"_{suffix}.pdf"
            )

        plot_spec(X_int, "int")
        plot_spec(X_orig, "orig")

        X_int = X_int[..., None]
        xhat_tm = Interpreter.invert_stft_with_phase(X_int, X_stft_phase).cpu()

        torchaudio.save(
            ".".join(hparams["single_sample"].split(".")[:-1]) + "_int.wav",
            xhat_tm,
            hparams["sample_rate"],
        )
