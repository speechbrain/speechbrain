import torch
import torch.nn as nn
import numpy as np
import json
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, DCT
from dcase_util.data import DecisionEncoder


def find_window_sizes(json_file, depooling, beta=0.333):
    """Find the average duration for each class from json file
        the subset should have strongly annotation.
        Arguments
        ---------
        json_file : string
            path to the json file.
        depooling : float
            to pass from frame to sec. ( ~pooling_time_ratio~ / ( ~sample_rate~ / ~hop_length~ ) )
        beta : float
            constant to adjust window sizes.

        Returns
        -------
        list of int
            window sizes for each class
    """
    with open(json_file) as js_file:
        data = json.load(js_file)
        class_dict = {"Alarm_bell_ringing":[],
                      "Speech":[],
                      "Dog":[],
                      "Cat":[],
                      "Vacuum_cleaner":[],
                      "Dishes":[],
                      "Frying":[],
                      "Electric_shaver_toothbrush":[],
                      "Blender":[],
                      "Running_water":[]
                      }
        # print(len(data))
        for sample_idx in range(len(data)):
            for key, event in data[f"{sample_idx}"]["event_label"].items():
                label = event[-1]
                class_dict[label].append(float(event[1])-float(event[0]))
        for k, v in class_dict.items():
            class_dict[k] = int((sum(v)/len(v))*(beta)/depooling)
            # print(k, sum(v)/len(v))

    list_mean= []
    for label in class_dict.keys():
            list_mean.append(class_dict[label])
    return list_mean

def update_ema_variables(model, ema_model, alpha, global_step):
    """from Curious AI https://arxiv.org/abs/1703.01780
        Update of weights from a model to another with small ramp-up at beginning.
        Arguments
        ---------
        model : torch.nn.Module
            Student model.
        ema_model : torch.nn.Module
            Teacher model.
        alpha : float
            constant in exp moving avg.
        global_step : integer
            step in the training

        Returns
        -------
        None
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242
        Arguments
        ---------
        current_step : integer
            step in the training
        rampup_length : integer
            measure to ramp-up with ease --> len(train_loader) * number_of_epochs // 2
        
        Returns
        -------
        float
            ramp_up value
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class ManyHotEncoder:
    """"
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py
        This class 4 main function:
            - encode weak (string to tensor)
            - encode strong (string to tensor)
            - decode weak (tensor to string)
            - decode strong (tensor to string)
    """
    def __init__(self, labels, max_length=10, n_frames=None):
        self.labels = labels
        self.n_frames = n_frames
        self.max_length = max_length

    def encode_weak(self, labels):
        """Encoder from string to tensor
        Arguments
        ---------
        labels : list, list of labels to encode
            step in the training
        
        Returns
        -------
        torch.tensor
            weakly annotation in a torch.tensor of size (n_class, )
        """

        y = torch.zeros(len(self.labels))

        if len(labels) > 1:
            for i in range(len(labels)):
                indx = self.labels.index(labels[i])
                y[indx]=1
        else:
            for i in labels:
                if i == 'unknown':
                    return y
                else:
                    indx = self.labels.index(i)
                    y[indx]=1
        return y

    def encode_strong(self, event_label_):
        """Encoder from string to tensor
        Arguments
        ---------
        event_label_ : dict of lists
            dict where each key is a different event in the sample (see example)
        
        Returns
        -------
        torch.tensor
            weakly annotation in a torch.tensor of size (n_class, )

        Input Example
        -------------

        {   "event_label_0":    [  "1.0690333475685596",
                                    "3.3843393475685595",
                                    "Alarm_bell_ringing"
                                ], 
            "event_label_1":    [  "3.886419868139693",
                                    "10.0",
                                    "Alarm_bell_ringing"
                                ] ...
        }
        """
        
        y = torch.zeros((self.n_frames, len(self.labels)))
        for key,value in event_label_.items():
            label = self.encode_weak([value[-1]])
            onset = int(float(value[0])*self.n_frames//self.max_length)
            offset = int(float(value[1])*self.n_frames//self.max_length)
            y[onset:offset] += label

        return y

    def decode_weak(self, labels):
        """ Decoder from tesnor to list of strings
         Arguments
        ---------
        labels : torch.tensor
            torch.tensor of encoded label
        
        Returns
        -------
        list
            list containing one or more events

        """
        result_labels = []
        indices = (labels == 1).nonzero().squeeze(-1).numpy()
        for idx in indices:
                result_labels.append(self.labels[int(idx)])
        return result_labels

    def decode_strong(self, encoded_labels, filename, depooling, append_list=False):
        """ Decoder from tensor to string (strong annotation)
             
         Arguments
        ---------
        encoded_labels : torch.tensor
            torch.tensor of encoded labels
        filename : string
            name of the file
        depooling : float
            to pass from frame to sec. ( ~pooling_time_ratio~ / ( ~sample_rate~ / ~hop_length~ ) )
        append_list : list
            list to append the return

        Returns
        -------
        list of dict
            list of the decoded annotation (see example)

        Output Example
        --------------
        [
            {
                'event_label': 'Dog',
                'onset': 0.00,
                'offset': 1.30,
                'filename': '1001.wav',
            },
            ...
            {
                'event_label': 'Speech',
                'onset': 2.22,
                'offset': 4.44,
                'filename': '1001.wav',
            },
        ]  
        """

        if not append_list:
            result_labels=[]
        for i, label_column in enumerate(encoded_labels.T):
            event_dict = {}
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)

            # append [label, onset, offset] in the result list
            for row in change_indices:
                event_dict = {}
                event_dict["event_label"] = self.labels[i]
                event_dict["onset"] = row[0]*depooling
                event_dict["offset"] = row[1]*depooling
                event_dict["filename"] = filename
                if not append_list:
                    result_labels.append(event_dict)
                else:
                    append_list.append(event_dict)
        if not append_list:
            return result_labels
        else:
            return append_list

def extract_ground_truth_list(event_label, filename):
    """ Function to transform information in json to a list
        to be used with SED eval toolkit.
             
         Arguments
        ---------
        event_label : dict
            see input of !ref <encode_strong>
        filename : string
            name of the file

        Returns
        -------
        list of dict
            list of the decoded annotation (see example)
        
        Output Example
        --------------
        [
            {
                'event_label': 'Dog',
                'onset': 0.00,
                'offset': 1.30,
                'filename': '1001.wav',
            },
            ...
            {
                'event_label': 'Speech',
                'onset': 2.22,
                'offset': 4.44,
                'filename': '1001.wav',
            },
        ]  
    """
    event_list = []
    for event, info in event_label.items():
            ev_dict={"event_label":"",
                     "onset":"",
                     "offset":"",
                     "filename":""}
            ev_dict["event_label"] = info[2]
            ev_dict["onset"] = info[0]
            ev_dict["offset"] = info[1]
            ev_dict["filename"] = filename
            event_list.append(ev_dict)
    return event_list

def trunc_seq(audio, max_len):
    """truncate too long sequence
        Arguments
        ---------
        audio : torch.tensor
            audio signal
        max_len : float or int
            maximum length in second
        
        Returns
        -------
        torch.tensor
            truncated audio or the same as input
    """
    length = audio.shape[0]

    if length > max_len:
        new_audio = audio[0:int(max_len)]
    else:
        new_audio = audio
    return new_audio

def to_mono(audio):
    """Transform signal that might be stereo to mono
        Arguments
        ---------
        audio : torch.tensor
            audio signal
        
        Returns
        -------
        torch.tensor
            audio signal mono (1-dimension)
    """
    if audio.ndimension() > 1:
        audio = torch.mean(audio, axis=1)
    return audio

def signal(wav, max_length):
    """Read audio + some basic adjustments
        Arguments
        ---------
        wav : string
            path to wav file
        
        Returns
        -------
        torch.tensor
            audio signal
    """
    sig = read_audio(wav)
    sig = to_mono(sig)
    sig = trunc_seq(sig, max_length)
    return sig

class ComputeFbank(torch.nn.Module):
    """Class to compute FBanks
        **should change it to Fbank module of SB**
    """
    def __init__(self, stft_args, fb_args):
        super().__init__()
        self.stft_args = stft_args
        self.fb_args = fb_args
    def forward(self, sig):
        compute_stft = STFT(**self.stft_args)
        stft = compute_stft(sig)
        spectral = spectral_magnitude(stft, power=0.5)
        compute_fbank = Filterbank(**self.fb_args)
        fb = compute_fbank(spectral)
        return fb

