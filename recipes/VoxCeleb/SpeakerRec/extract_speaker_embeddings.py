#!/usr/bin/python3
"""Recipe for extracting speaker embeddings for other purpose. This
is more like a script that copes with modern usage of speaker embed-
ding vectors.

The input of this script is a training list like below
(we recommend having full absolute path for wav paths)
----------
utt1 $wav1_path
...
uttN $wavN_path

The extracted embeddings are stored as numpy files in the output
folder. The name of each numpy file is its utterance name.
NOTE: This may result in a large number of files in a single folder.

To run this recipe, use the following command:
> python extract_speaker_embeddings.py {input_training_list} {output_folder} {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hparams/verification_ecapa.yaml (for the ecapa+tdnn system)
    hparams/verification_resnet.yaml (for the resnet tdnn system)
    hparams/verification_plda_xvector.yaml (for the xvector system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
    * Xuechen Liu 2023
"""
import os
import sys

import numpy as np
import torch
import logging
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import download_file


def compute_embeddings_single(wavs, wav_lens, params):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embeddings(params, wav_scp, outdir):
    """Compute speaker embeddings.

    Arguments
    ---------
    params: dict
        The parameter files storing info about model, data, etc
    wav_scp : str
        The wav.scp file in Kaldi, in the form of "$utt $wav_path"
    outdir: str
        The output directory where we store the embeddings in per-
        numpy manner.
    """
    with torch.no_grad():
        with open(wav_scp, "r") as wavscp:
            for line in wavscp:
                utt, wav_path = line.split()
                out_file = "{}/{}.npy".format(outdir, utt)
                wav, _ = torchaudio.load(wav_path)
                data = wav.transpose(0, 1).squeeze(1).unsqueeze(0)
                lens = torch.Tensor([data.shape[1]])
                data, lens = (
                    data.to(run_opts["device"]),
                    lens.to(run_opts["device"]),
                )
                embedding = compute_embeddings_single(
                    data, lens, params
                ).squeeze()

                out_embedding = embedding.detach().cpu().numpy()
                np.save(out_file, out_embedding)
                del out_embedding, wav, data


if __name__ == "__main__":
    in_list = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[3:])
    if "data_folder:" not in overrides:
        # By default it is a PLACEHOLDER (we need to replace it with a dummy path)
        overrides += "\ndata_folder: ."
    if "output_folder:" not in overrides:
        # Ensure to put the saved model in the output folder
        overrides += f"\noutput_folder: {out_dir}"

    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(run_opts["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    download_file(params["verification_file"], veri_file_path)

    print("Begin embedding extraction......")
    compute_embeddings(params, in_list, out_dir)
    print("The embeddings have been extracted and stored at {}".format(out_dir))
