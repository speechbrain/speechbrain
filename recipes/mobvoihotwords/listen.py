import argparse
import os
import random
from datetime import datetime
from pprint import pprint

import torch
import numpy as np

# from utils.file_utils import common_path

import sys

# from run.run_utils import merge_configs, init_data_loader, set_seed

# from utils import Workspace, find_cls, load_json, prepare_device, num_floats_to_GB
# import librosa
# from utils import AudioProcessor, get_feature

import matplotlib.pyplot as plt

import time
# from utils.file_utils import common_path
from listener.realtime_processing import realtime_processing
from listener.TriggerDetector import TriggerDetector

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from train import SpeakerBrain
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
import torchaudio
torchaudio.set_audio_backend("soundfile")


class Listener(realtime_processing):
    
    def __init__(self, model, feature_type, Recording=False):
        super(Listener, self).__init__(model, feature_type, Recording=Recording)
        # self.audio_processor = AudioProcessor()
        self.model = model
        self.model.eval()
        # self.audio_processor = AudioProcessor()
        self.feature_type = feature_type

        self.hop_len = 160
        self.win_len = 480
        self.CHUNK = 960
        self.overlap = self.win_len - self.hop_len   # 320
        self.data_buffer = np.zeros((self.CHUNK + self.overlap,)) # 1280

        # self.input = np.zeros((148,40), dtype=np.float32)
        self.frame_num = 151
        self.feat_dim = 40
        self.input = np.random.random((1, self.frame_num, self.feat_dim))
        self.input = np.array(self.input).astype(np.float32)

        self.new_frame_num = 6

        self.postprocessing = TriggerDetector(self.CHUNK, sensitivity=0.5, trigger_level=3)
        self.wake_count = 0

        self.keyword = 0

    def process(self, data):

        self.data_buffer[:self.overlap] = self.data_buffer[-self.overlap:]
        self.data_buffer[-self.CHUNK:] = data

        # print(self.data_buffer.dtype)

        # return 0
        # feat = torch.rand((1, 151, 40)).to(se_brain.device)
        feat = np.random.random((1, self.new_frame_num, self.feat_dim))
        noisy_feats = se_brain.modules.compute_features(torch.from_numpy(self.data_buffer[np.newaxis, :].astype(np.float32)))
        # feat = get_feature(self.data_buffer, audio_preprocessing=self.feature_type, audio_processor=self.audio_processor)
        self.input[:, :(self.frame_num - self.new_frame_num), :] = self.input[:, -1*((self.frame_num - self.new_frame_num)):, :]
        self.input[:, -1*self.new_frame_num:, :] = feat
        # print("feat :{}".format(feat.shape))
        # total_loss = 0
        net_input = self.input[:, np.newaxis, :, :]
        # print("data:{}".format(net_input.shape))

        data_tensor = torch.from_numpy(net_input[0, ...])
        data_tensor[:, -1*self.new_frame_num:, :] = noisy_feats.detach().clone()
        data_tensor = se_brain.modules.mean_var_norm(data_tensor, torch.ones([1]).to(se_brain.device))

        output = []
        output_0 = []
        output_1 = []

        prob = 0.0

        with torch.no_grad():
            for i in range(data_tensor.shape[0]):
                # print("data_tensor[i, :, :, :].shape.{}".format(data_tensor[i:i+1, :, :, :].shape))
                # print(data_tensor[i:i+1, :, :].shape)
                y = self.model(data_tensor[i:i+1, :, :])
                y = y[0, 0, :]
                
                if torch.argmax(y) == self.keyword:
                    print("prob:{}".format(y))

                y = np.exp(y.detach().cpu().numpy())
                prob = y[self.keyword]
                
                if self.postprocessing.update(prob):
                    self.wake_count = self.wake_count + 1
                    print("{}:.................keyword detected!..............................".format(self.wake_count))
                # output_0.append(y.cpu().numpy()[0, 0])
                # output_1.append(y.cpu().numpy()[0, 1])

        line_with = 1   # [1, self.CHUNK]
        self.audio_data[:self.buffer_len - line_with, 0] = self.audio_data[line_with:, 0]
        self.audio_data[-line_with:, 0] = prob * np.ones((line_with,))

        return output


def chop_array(arr, window_size, hop_size):
    """chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]"""
    return [arr[:, i - window_size:i, :] for i in range(window_size, arr.shape[1] + 1, hop_size)]


def power_spec(audio: np.ndarray, window_stride=(160, 80), fft_size=512):
    """Calculates power spectrogram"""
    frames = chop_array(audio, *window_stride) or np.empty((0, window_stride[0]))
    fft = np.fft.rfft(frames, n=fft_size)
    return (fft.real ** 2 + fft.imag ** 2) / fft_size


def main(se_brain):
    model = se_brain.modules.embedding_model


    listener = Listener(model, feature_type='FBANK')


    listener.start()



if __name__ == "__main__":
    print(sys.path)

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

    se_brain.on_evaluate_start(min_key="ErrorRate")
    se_brain.on_stage_start(sb.Stage.TEST, epoch=None)

    se_brain.modules.eval()

    main(se_brain)
