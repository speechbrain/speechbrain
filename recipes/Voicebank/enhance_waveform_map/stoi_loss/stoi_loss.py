# ################################
# From paper: "End-to-End Waveform Utterance Enhancement for Direct Evaluation
# Metrics Optimization by Fully Convolutional Neural Networks", TASLP, 2018
# Authors: Szu-Wei, Fu 2020
# ################################

import torch
import torchaudio
import numpy as np


def thirdoct(fs, nfft, num_bands, min_freq):
    """ Returns the 1/3 octave band matrix
    # Arguments :
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    # Returns :
        obm : Octave Band Matrix
    """
    f = torch.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = torch.from_numpy(np.array(range(num_bands)).astype(float))
    cf = torch.pow(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * torch.pow(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * torch.pow(2.0, (2 * k + 1) / 6)
    obm = torch.zeros((num_bands, len(f)))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = torch.argmin(torch.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = torch.argmin(torch.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm


def removeSilentFrames(x, y, dyn_range=40, N=256, K=128):
    w = torch.from_numpy(np.hanning(257))
    frames = range(0, x.shape[0] - N, K)
    energy = []
    for frame in frames:
        energy.append(
            20 * torch.log10(torch.norm(w * x[frame : (frame + N)]) / 16.0)
        )

    msk = torch.zeros(len(frames))
    Max_energy = max(energy)
    i = 0
    for e in energy:
        if e - Max_energy + dyn_range > 0:
            msk[i] = 1
        i = i + 1

    x_sil = torch.zeros(x.shape)
    y_sil = torch.zeros(y.shape)
    count = 0
    for j in range(len(frames)):
        if msk[j] == 1:
            x_sil[frames[count] : frames[count] + N] = (
                x_sil[frames[count] : frames[count] + N]
                + w * x[frames[j] : frames[j] + N]
            )
            y_sil[frames[count] : frames[count] + N] = (
                y_sil[frames[count] : frames[count] + N]
                + w * y[frames[j] : frames[j] + N]
            )
            count = count + 1

    return [x_sil[0 : frames[count - 1] + N], y_sil[0 : frames[count - 1] + N]]


def stoi_loss(y_pred_batch, y_true_batch, lens):
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    y_pred_batch_shape = y_pred_batch.shape

    batch_size = y_pred_batch_shape[0]

    fs = 16000  # Sampling rate
    N = 30  # length of temporal envelope vectors
    J = 15.0  # Number of one-third octave bands
    smallVal = 0.0000000001  # To avoid divide by zero

    octave_band = thirdoct(fs=10000, nfft=512, num_bands=15, min_freq=150)
    c = 5.62341325  # 10^(-Beta/20) with Beta = -15
    D = 0
    for i in range(0, batch_size):  # Run over mini-batches
        y_true = y_true_batch[i, 0 : int(lens[i] * y_pred_batch_shape[1])]
        y_pred = y_pred_batch[i, 0 : int(lens[i] * y_pred_batch_shape[1])]

        y_true = torchaudio.transforms.Resample(fs, 10000)(y_true)
        y_pred = torchaudio.transforms.Resample(fs, 10000)(y_pred)

        [y_sil_true, y_sil_pred] = removeSilentFrames(y_true, y_pred)

        stft_true = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=256, hop_length=128, power=2
        )(y_sil_true)
        stft_pred = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=256, hop_length=128, power=2
        )(y_sil_pred)

        OCT_true = torch.sqrt(torch.matmul(octave_band, stft_true))
        OCT_pred = torch.sqrt(torch.matmul(octave_band, stft_pred))

        M = int(
            stft_pred.shape[-1] - (N - 1)
        )  # number of temporal envelope vectors

        d = 0
        for m in range(0, M):  # Run over temporal envelope vectors
            x = OCT_true[:, m : m + N]
            y = OCT_pred[:, m : m + N]

            alpha = torch.sqrt(
                torch.sum(torch.square(x), dim=-1, keepdim=True)
                / (torch.sum(torch.square(y), dim=-1, keepdim=True) + smallVal)
            )

            ay = y * alpha
            y = torch.min(ay, x + x * c)

            xn = x - torch.mean(x, dim=-1, keepdim=True)
            xn = xn / (
                torch.sqrt(torch.sum(xn * xn, dim=-1, keepdim=True)) + smallVal
            )

            yn = y - torch.mean(y, dim=-1, keepdim=True)
            yn = yn / (
                torch.sqrt(torch.sum(yn * yn, dim=-1, keepdim=True)) + smallVal
            )

            di = torch.sum(xn * yn, dim=-1, keepdim=True)
            d = d + torch.sum(di, dim=0, keepdim=False)
        D = D + d / (J * M)

    return -(D / (batch_size))
