"""Basic feature pipelines.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import numpy as np
import torch
import pysptk
import librosa
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    Filterbank,
    DCT,
    Deltas,
    ContextWindow,
)


class DecoderLabeller(torch.nn.Module):
    def forward(self, wav):
        """Returns the label for the waveform decoder.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        return wav.unsqueeze(2)


class LPSLabeller(torch.nn.Module):
    def forward(self, compute_STFT,wav):
        feats = compute_STFT(wav)
        feats = spectral_magnitude(feats, power=1)# power spectrum

        # Log1p reduces the emphasis on small differences
        feats = torch.log1p(feats)

        return feats

class ProsodyLabeller(torch.nn.Module):

    def __init__(self,
                 hop=160,
                 win=320,
                 f0_min=60,
                 f0_max=300,
                 sr=16000
           ):
           self.hop = hop
           self.win = win
           self.f0_min = f0_min
           self.f0_max= f0_max
           self.sr = sr
           super().__init__()

    def linear_interpolation(self, tbounds, fbounds):
        """Linear interpolation between the specified bounds"""
        interp = []
        for t in range(tbounds[0], tbounds[1]):
            interp.append(fbounds[0] + (t - tbounds[0]) * ((fbounds[1] - fbounds[0]) /
                                                           (tbounds[1] - tbounds[0])))
        return interp

    def interpolation(self, signal, unvoiced_symbol):
        tbound = [None, None]
        fbound = [None, None]
        signal_t_1 = signal[0]
        isignal = np.copy(signal)
        uv = np.ones(signal.shape, dtype=np.int8)
        for t in range(1, signal.shape[0]):
            if (signal[t] > unvoiced_symbol) and (signal_t_1 <= unvoiced_symbol) and (tbound == [None, None]):
                # First part of signal is unvoiced, set to constant first voiced
                isignal[:t] = signal[t]
                uv[:t] = 0
            elif (signal[t] <= unvoiced_symbol) and (signal_t_1 > unvoiced_symbol):
                tbound[0] = t - 1
                fbound[0] = signal_t_1
            elif (signal[t] > unvoiced_symbol) and (signal_t_1 <= unvoiced_symbol):
                tbound[1] = t
                fbound[1] = signal[t]
                isignal[tbound[0]:tbound[1]] = self.linear_interpolation(tbound, fbound)
                uv[tbound[0]:tbound[1]] = 0
                # reset values
                tbound = [None, None]
                fbound = [None, None]
            signal_t_1 = signal[t]
        # now end of signal if necessary
        if tbound[0] is not None:
            isignal[tbound[0]:] = fbound[0]
            uv[tbound[0]:] = 0
        # if all are unvoiced symbols, uv is zeros
        if np.all(isignal <= unvoiced_symbol):
            uv = np.zeros(signal.shape, dtype=np.int8)
        return isignal, uv

    def cal_prosody(self,wav):
        # Input: wav: audio signal in numpy.array format
        # Output: proso: Tensor, [max_frames, 4]
        max_frames = wav.shape[0] // self.hop

        f0 = pysptk.swipe(wav.astype(np.float64),
                          fs=self.sr, hopsize=self.hop,
                          min=self.f0_min,
                          max=self.f0_max,
                          otype='f0')
        lf0 = np.log(f0 + 1e-10)
        lf0, uv = self.interpolation(lf0, -1)

        lf0 = torch.tensor(lf0.astype(np.float32)).unsqueeze(0)[:, :max_frames]# (1,num_frame)
        uv = torch.tensor(uv.astype(np.float32)).unsqueeze(0)[:, :max_frames]
        if torch.sum(uv) == 0:
            # if frame is completely unvoiced, make lf0 min val
            lf0 = torch.ones(uv.size()) * np.log(self.f0_min)
        assert lf0.min() > 0, lf0.data.numpy()
        # secondly obtain zcr
        zcr = librosa.feature.zero_crossing_rate(y=wav,
                                                 frame_length=self.win,
                                                 hop_length=self.hop)
        zcr = torch.tensor(zcr.astype(np.float32))
        zcr = zcr[:, :max_frames]
        # finally obtain energy
        egy = librosa.feature.rms(y=wav, frame_length=self.win,
                                   hop_length=self.hop,
                                   pad_mode='constant')
        egy = torch.tensor(egy.astype(np.float32))
        egy = egy[:, :max_frames]
        proso = torch.cat((lf0, uv, egy, zcr), dim=0).unsqueeze(0)#(1,4,num_frame)
        return proso

    def forward(self,wav):

        prosody_list = [self.cal_prosody(ex.cpu().numpy()) for ex in wav]
        prosody_feats = torch.cat(prosody_list, dim = 0).transpose(1,-1).to(wav.device)#(batch_size,num_frame,4)

        return prosody_feats


class LIMLabeller(torch.nn.Module):
    def forward(self, pred):
        bsz, slen = pred.size(0) // 2, pred.size(1)

        return torch.cat((
            torch.ones(bsz, slen, 1, requires_grad=False),
            torch.zeros(bsz, slen, 1, requires_grad=False)
            ),
            dim=0,
        )

class GIMLabeller(torch.nn.Module):
    def forward(self, pred):
        bsz, slen = pred.size(0) // 2, pred.size(1)

        return torch.cat((
            torch.ones(bsz, slen, 1, requires_grad=False),
            torch.zeros(bsz, slen, 1, requires_grad=False)
            ),
            dim=0,
        )

class SPCLabeller(torch.nn.Module):
    def forward(self, pred):
        bsz, slen = pred.size(0) // 2, pred.size(1)

        return torch.cat((
            torch.ones(bsz, slen, 1, requires_grad=False),
            torch.zeros(bsz, slen, 1, requires_grad=False)
            ),
            dim=0,
        )
