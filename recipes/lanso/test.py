#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/mini_librispeech

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
"""
import sys
import os
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from train import SpeakerBrain
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile

def save_audio(filename: str, audio: np.ndarray, fs=16000):
    """Save loaded audio to file using the configured audio parameters"""
    if not filename.endswith(".wav"):
        filename = filename + ".wav"
    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

    wavfile.write(filename, fs, audio)   # audio should be (Nsamples, Nchannels)

def save_prob_to_wav(audio: np.ndarray, prob_vec: np.ndarray, hop_len: int, predict_win: int, filename: str, delay=0):
    """combine mono audio with prob_vec to stereo data

    Args:
        audio (np.ndarray): [description]
        prob_vec (np.ndarray): [description]
        win_len (int): [description]
        filename (str): [description]
        delay (int, optional): [description]. Defaults to 0.
    """
    from scipy.io import wavfile
    audio = np.squeeze(audio)
    output_wavdata = np.zeros((len(audio), 2))
    output_wavdata[:, 0] = np.squeeze(audio)
    frame_num = int(len(audio)/hop_len) - predict_win

    print("frame_len:{}".format(frame_num))
    print("frame_len:{}".format(frame_num))

    prob_interp = np.zeros(len(audio))
    print(audio.shape)
    print(prob_vec.shape)
    print(prob_interp.shape)

    for n in range(frame_num-2):
        prob_interp[n*hop_len:(n+1)*hop_len] = np.ones(hop_len) * prob_vec[n]

    if delay >0:   # delay channel-2
        output_wavdata[delay:, 1] = prob_interp[:len(output_wavdata[delay:, 1])]
    else:          # delay channel-2
        output_wavdata[:len(prob_interp[delay*-1:]), 1] = prob_interp[delay*-1:]

    save_audio(filename, output_wavdata)


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    print(hparams_file)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    se_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file
    )

    ind2lab = label_encoder.ind2lab
    print(ind2lab)
    lab2ind = label_encoder.lab2ind
    print(lab2ind)

    words_wanted=['小蓝小蓝', '管家管家', '物业物业', 'unknown']


    wav = 'wav/xmos/hixiaowen/音轨-4.wav'
    # wav = 'wav/diy/xiaolanxiaolan/1/音轨-2.wav'
    # wav = 'wav/hixiaowen_unk_nihaowenwen2.wav'
    wav = 'wav/xmos/noise1/音轨-4.wav'

    wav_data = sb.dataio.dataio.read_audio(wav)
    noisy_wavs = wav_data.reshape(1, -1)
    print("noisy_wavs.shape:{}".format(noisy_wavs.shape))

    noisy_feats = se_brain.modules.compute_features(noisy_wavs)
    print("noisy_feats.shape:{}".format(noisy_feats.shape))

    se_brain.on_evaluate_start(min_key="ErrorRate")
    se_brain.on_stage_start(sb.Stage.TEST, epoch=None)

    print("Epoch loaded: {}".format(hparams['epoch_counter'].current))

    se_brain.modules.eval()

    noisy_feats = noisy_feats.to(se_brain.device)
    noisy_wavs = noisy_wavs.to(se_brain.device)
    lens = torch.tensor(noisy_feats.shape[1])

    # noisy_feats = se_brain.modules.mean_var_norm(noisy_feats, lens)
    # print("feats.size():{}".format(feats.size()))

    # Embeddings + classifier
    predict_win = 151
    frame_num = noisy_feats.shape[1]
    output_wavdata = np.zeros([noisy_wavs.shape[1], 2])

    output_list = []
    output_list = [{} for i in range(len(words_wanted))]
    for i, word in enumerate(words_wanted):
        output_list[i][word] = []

    output_unk_list = []
    for n in range(frame_num - predict_win):
        feat_n = noisy_feats[:, n:n+predict_win, :]#.detach().clone()
        feats = feat_n.detach().clone()
        feats = se_brain.modules.mean_var_norm(feats, torch.ones([1]).to(se_brain.device))

        output = se_brain.modules.embedding_model(feats)
        if "classifier" in se_brain.modules.keys():
            output = se_brain.modules.classifier(output)

        output = np.exp(output.detach().cpu().numpy()[0,0,:])

        if np.argmax(output) != lab2ind['unknown']:
            print('{}:{}'.format(ind2lab[np.argmax(output)], np.max(output)))

        for i, word in enumerate(words_wanted):
            output_list[i][word].append(output[lab2ind[word]])

    for i, word in enumerate(words_wanted):
        prob = np.array(output_list[i][word])
        plt.figure()
        plt.plot(prob)
        plt.savefig('wav/output/' + word + '.png')

        if word == 'unknown':
            hop_len = 160
            save_prob_to_wav(wav_data, 1-prob, hop_len, predict_win, 'wav/output/hixiaowen_unk_nihaowenwen2_prob3.wav', delay=8000)
