"""Recipe for extract speaker embeddings from a kaldi-formed directory with:
1. wav.scp, in terms of [utt, wav_path] for each line
2. utt2spk

>  python speaker_inference_kaldi.py [opts] hyperparams/verification_plda_xvector.yaml

Authors
    * Xuechen Liu 2021
"""

import argparse
import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
import numpy as np
import kaldi_io
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.distributed import run_on_main


def write_vecs_to_kaldi(vec, utt, ark_reader):
    """write vectors to kaldi-format archive reader, for further kaldi processing
    if necessary
    """
    try:
        vec_dim = vec.shape
    except ValueError:
        raise Exception("failed getting vector stats from {0}".format(utt))
    # print("writing vec in terms of kaldi arks for utterance {0} with {1} dims...".format(utt, vec_dim))
    try:
        kaldi_io.write_vec_flt(ark_reader, vec, key=utt)
    except:
        raise Exception("vector writing error for {0}".format(utt))


def copy_vecs_kaldi(ark_file, tar_ark, tar_scp):
    command = "copy-vector ark:{0} ark,scp:{1},{2}".format(
        ark_file, tar_ark, tar_scp
    )
    os.system(command)


# Compute embeddings from the waveforms
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
    wavs = wavs.to(params["device"])
    wav_lens = wav_lens.to(params["device"])
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens).to(params["device"])
        embeddings = params["embedding_model"](feats)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)


def compute_embeddings(params, wav_scp, outdir, ark):
    with torch.no_grad():
        with open(wav_scp, "r") as wavscp:
            for line in wavscp:
                utt, wav_path = line.split()
                out_file = "{}/npys/{}.npy".format(outdir, utt)
                wav, _ = torchaudio.load(wav_path)
                data = wav.transpose(0, 1).squeeze(1).unsqueeze(0)
                embedding = compute_embeddings_single(data, torch.Tensor([data.shape[0]]), params).squeeze()

                out_embedding = embedding.detach().cpu().numpy()
                np.save(out_file, out_embedding)
                write_vecs_to_kaldi(out_embedding, utt, ark)
                del out_embedding, wav, data


if __name__ == "__main__":

    # load directories
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--datadir", type=str, required=True)
    #parser.add_argument("--outdir", type=str, required=True)

    #args = parser.parse_args()
    #datadir = args.datadir
    #outdir = args.outdir
    datadir = sys.argv[1]
    outdir = sys.argv[2]

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[3:])
    print(params_file)
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides=None)
    
    # Load model
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # perform embedding extraction
    os.makedirs(outdir + "/npys", exist_ok=True)
    ark_file = outdir + "/raw_xvector.ark"

    print("begin embedding extraction......")
    with open(ark_file, "wb") as ark:
        compute_embeddings(params, datadir + "/wav.scp", outdir, ark)

    copied_ark_file = outdir + "/xvector.ark"
    copied_scp_file = outdir + "/xvector.scp"
    copy_vecs_kaldi(ark_file, copied_ark_file, copied_scp_file)
    print(
        "X-Vectors extracted from {0} has been stored in {1}".format(
            datadir, outdir
        )
    )
