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
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import DynamicItemDataset

from train import SpeakerBrain
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
from tqdm import tqdm

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

def compute_metrics(result : dict, verbose=True):
    print("result for hixiaowen")

    TP = result['TP']
    FN = result['FN']
    TN = result['TN']
    FP = result['FP']

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    false_positive_rate = FP / (FP + TN) if FP + TN > 0 else 0.0
    false_negative_rate = FN / (FN + TP) if FN + TP > 0 else 0.0

    if verbose:
        print("True Positive:{}".format(result['TP']))
        print("False Negative:{}".foramt(result['FN']))
        print("True Negative:{}".foramt(result['TN']))
        print("False Positive:{}".foramt(result['FP']))
        print("precise:{}".format(precision))
        print("recall:{}".format(recall))
        print("false_positive_rate:{}".format(false_positive_rate))
        print("false_negative_rate:{}".format(false_negative_rate))

    return precision, recall, false_positive_rate, false_negative_rate


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("signal")
def read_audio(wav):
    sig = sb.dataio.dataio.read_audio(wav)
    return sig

from speechbrain.dataio.dataio import load_data_json, load_data_csv
def eval_func():

    data_csv = load_data_csv("results/save/test.csv")
    print(len(data_csv))
    print(data_csv[list(data_csv.keys())[0]].keys())

    dataset = DynamicItemDataset.from_csv("results/save/test.csv")
    dataset.add_dynamic_item(sb.dataio.dataio.read_audio, takes="wav", provides="signal") 
    dataset.set_output_keys(['wav', 'command', 'signal'])

    for n in range(len(dataset)):
        data = dataset[n]
        print('id:{}'.format(data['wav']))
        print('len:{}'.format(len(data['signal'])))
        print('command:{}'.format(data['command']))

    return


# Recipe begins!
if __name__ == "__main__":

    # eval_func()


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

    # wav = 'wav/xmos/hixiaowen/音轨-4.wav'
    # # wav = 'wav/hixiaowen_unk_nihaowenwen2.wav'

    # wav_data = sb.dataio.dataio.read_audio(wav)
    # noisy_wavs = wav_data.reshape(1, -1)
    # print("noisy_wavs.shape:{}".format(noisy_wavs.shape))

    # noisy_feats = se_brain.modules.compute_features(noisy_wavs)
    # print("noisy_feats.shape:{}".format(noisy_feats.shape))

    se_brain.on_evaluate_start(min_key="ErrorRate")
    se_brain.on_stage_start(sb.Stage.TEST, epoch=None)

    print("Epoch loaded: {}".format(hparams['epoch_counter'].current))

    se_brain.modules.eval()

    # noisy_feats = noisy_feats.to(se_brain.device)
    # noisy_wavs = noisy_wavs.to(se_brain.device)
    # lens = torch.tensor(noisy_feats.shape[1])

    predict_win = 151

    # noisy_feats = se_brain.modules.mean_var_norm(noisy_feats, lens)
    # print("feats.size():{}".format(feats.size()))

    dataset = DynamicItemDataset.from_csv("results/save/dev.csv")
    dataset.add_dynamic_item(sb.dataio.dataio.read_audio, takes="wav", provides="signal")
    dataset.set_output_keys(['wav', 'command', 'signal'])

    result  = {}
    target_label = {'hixiaowen': 0, 'nihaowenwen': 1, 'unknown': 2}

    for wake_word in ['hixiaowen', 'nihaowenwen']:
        result[wake_word] = {}
        result[wake_word].update({'TP': 0})
        result[wake_word].update({'FN': 0})
        result[wake_word].update({'FP': 0})
        result[wake_word].update({'TN': 0})

    win_len = 24000

    for n in tqdm(range(len(dataset))):
        data = dataset[n]
        noisy_wavs = data['signal']
        if len(noisy_wavs) < win_len:
            noisy_wavs = torch.cat((noisy_wavs, torch.zeros(win_len - len(noisy_wavs))))
        noisy_wavs = noisy_wavs.reshape(1, -1)

        label = target_label[data["command"]]

        triger_count = 0

        noisy_feats = se_brain.modules.compute_features(noisy_wavs)
        noisy_feats = noisy_feats.to(se_brain.device)
        frame_num = noisy_feats.shape[1]
        # print(noisy_feats.shape)
        compute_cw  = sb.processing.features.ContextWindow(left_frames=75, right_frames=75)
        feats_contex = compute_cw(noisy_feats)
        # print("feats_contex:{}".format(feats_contex.shape))
        feats_contex = torch.reshape(feats_contex, (frame_num, 151, 40))
        # noisy_feats = torch.transpose(feats_contex, 0, 1)
        # print("feats_contex:{}".format(feats_contex.shape))
        # print(noisy_feats.shape)

        output_list = []
        # output_label = []

        # noisy_feats = torch.reshape(noisy_feats, (-1, 1, predict_win, 40))
        feats = se_brain.modules.mean_var_norm(feats_contex, torch.ones([frame_num]).to(se_brain.device))
        # print("feats:{}".format(feats.shape))
        output = se_brain.modules.embedding_model(feats)
        # print(output.shape)

        output_label = torch.argmax(output[:, 0, :], dim=1).cpu().numpy()
        # output_label = np.sum(output_label)
        # print("output_label:{}".format(output_label.shape))
        # print(len(output_label))

        keyword1_count = 0
        keyword2_count = 0

        for t in range(frame_num):
            if output_label[t] == 0:
                keyword1_count += 1
            if output_label[t] == 1:
                keyword2_count += 1

        if data["command"] == 'hixiaowen':
            if keyword1_count > 0:
                result['hixiaowen']['TP'] += 1
            else:
                result['hixiaowen']['FN'] += 1
            if keyword2_count > 0:
                result['nihaowenwen']['FP'] += 1
            else:
                result['nihaowenwen']['TN'] += 1
        if data["command"] == 'nihaowenwen':
            if keyword1_count > 0:
                result['hixiaowen']['FP'] += 1
            else:
                result['hixiaowen']['TN'] += 1
            if keyword2_count > 0:
                result['nihaowenwen']['TP'] += 1
            else:
                result['nihaowenwen']['FN'] += 1

        # # for t in range(frame_num - predict_win):
        # #     feat_t = noisy_feats[:, t:t+predict_win, :]
        # #     feats = se_brain.modules.mean_var_norm(feat_t, torch.ones([1]).to(se_brain.device))
        # #     output = se_brain.modules.embedding_model(feats)
        # #     # output = np.exp(output.detach().cpu().numpy()[0,0,:])

        # #     output_label.append(torch.argmax(output[0, 0, :]).cpu().numpy())
        # # output_count = np.sum(np.array(output_label))
        # if target_label[data["command"]] == 'hixiaowen':
        #     if output_count >= 1:
        #         result['hixiaowen']['TP'] += 1
        #     else:
        #         result['hixiaowen']['FN'] += 1
        # else:
        #     if output_label == 0:
        #         result['hixiaowen']['TN'] += 1
        #     else:
        #         result['hixiaowen']['FP'] += 1

    print("result for hixiaowen")
    compute_metrics(result['hixiaowen'])

    print("result for nihaowenwen")
    compute_metrics(result['nihaowenwen'])


        #     output_list.append(output[0])

        #     # for wake_word in ['hixiaowen', 'nihaowenwen']:

        #     #     result[wake_word].upd
        # output_prob = np.array(output_list)


    #     print('id:{}'.format(data['wav']))
    #     print('len:{}'.format(len(data['signal'])))
    #     print('command:{}'.format(data['command']))

    # # Embeddings + classifier
    # predict_win = 151
    # frame_num = noisy_feats.shape[1]
    # output_wavdata = np.zeros([noisy_wavs.shape[1], 2])

    # output_list = []
    # for n in range(frame_num - predict_win):
    #     feat_n = noisy_feats[:, n:n+predict_win, :]
    #     feats = se_brain.modules.mean_var_norm(feat_n, torch.ones([1]).to(se_brain.device))
    #     output = se_brain.modules.embedding_model(feats)
    #     # print(outputs.size())
    #     output = np.exp(output.detach().cpu().numpy()[0,0,:])
    #     # print(outputs.shape)

    #     output_list.append(output[0])



    # #     output_prob[n*win_len:(n+1)*win_len] = np.ones(win_len) * output[0]


    # # output_wavdata[:, 0] = wav_data             # # copy raw data to channel-1
    # #                                             # save prob to channel-2
    # # delay = 16000
    # # output_wavdata[delay:, 1] = output_prob[:len(output_wavdata[delay:, 1])]

    # # save_audio('wav/output/a3_prob.wav',output_wavdata)

    # output_prob = np.array(output_list)

    # hop_len = 160
    # save_prob_to_wav(wav_data, output_prob, hop_len, predict_win, 'wav/output/hixiaowen_unk_nihaowenwen2_prob2.wav', delay=8000)

    # print(output_prob.shape)
    # plt.figure()
    # plt.plot(output_prob)
    # plt.savefig('wav/output/output2.png')

