#!/usr/bin/env python3
"""Recipe for extract speaker embeddings from a kaldi-formed directory with:
1. wav.scp, in terms of [utt, wav_path] for each line
2. utt2spk

>  python speaker_inference_kaldi.py [opts] hyperparams/verification_plda_xvector.yaml

Authors
    * Xuechen Liu 2021

(specialized for household with two options
    1. perform trimming via librosa, 20dB
    2. 2-sec cropping
)

"""

import argparse
import librosa
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
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)


def compute_embeddings(params, wav_scp, outdir, ark, trimmed=False, chunk_length=0):

    def __get_random_chunk(wav, length):
        """given an utterance length utt_length (in frames) and two desired chunk lengths
        (length1 and length2) whose sum is <= utt_length,
        this function randomly picks the starting points of the chunks for you.
        the chunks may appear randomly in either order.
        """
        utt_length = wav.shape[0]
        if length >= utt_length:
            chunked_wav = wav
        else:
            free_length = utt_length - length
            offset = np.random.randint(0, free_length)
            chunked_wav = wav[offset:offset+length]
        return chunked_wav

    def __get_trimmed(wav):
        wav_samples = wav.numpy()
        trimmed_samples = librosa.effects.trim(wav_samples, top_db=20)
        return torch.from_numpy(trimmed_samples)


    with torch.no_grad():
        with open(wav_scp, "r") as wavscp:
            for line in wavscp:
                utt, wav_path = line.split()
                out_file = "{}/npys/{}.npy".format(outdir, utt)
                wav, _ = torchaudio.load(wav_path).transpose(0, 1).squeeze(1)
                if trimmed:
                    wav = __get_trimmed(wav)
                if chunk_length != 0:
                    wav = __get_random_chunk(wav, chunk_length)
                embedding = compute_embeddings_single(wav, wav.shape[0], params)

                out_embedding = embedding.detach().cpu().numpy()
                np.save(out_file, out_embedding)
                write_vecs_to_kaldi(out_embedding, utt, ark)
                del out_embedding, wav


if __name__ == "__main__":

    datadir = sys.argv[1]
    outdir = sys.argv[2]
    trimmed = bool(sys.argv[3]) # 0 means false
    chunk_length = int(sys.argv[4])

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[5:])
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
        compute_embeddings(params, datadir + "/wav.scp", outdir, ark, trimmed=False, chunk_length=0)

    copied_ark_file = outdir + "/xvector.ark"
    copied_scp_file = outdir + "/xvector.scp"
    copy_vecs_kaldi(ark_file, copied_ark_file, copied_scp_file)
    print(
        "X-Vectors extracted from {0} has been stored in {1}".format(
            datadir, outdir
        )
    )
