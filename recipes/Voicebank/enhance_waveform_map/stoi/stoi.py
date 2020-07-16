# ################################
# Authors: Szu-Wei, Fu 2020
# ################################

import numpy as np
from resampy import resample
from scipy import signal

N = 30  # length of temporal envelope vectors
J = 15.0  # Number of one-third octave bands
smallVal = np.finfo("float").eps  # To avoid divide by zero

c = 5.62341325  # 10^(-Beta/20) with Beta = -15


def thirdoct(fs, N_fft, numBands, mn):
    """ returns 1/3 octave band matrix
    inputs :
        FS:         samplerate
        N_fft:      FFT size
        numBands:   number of bands
        mn:         center frequency of first 1/3 octave band
    outputs :
        A : Octave Band Matrix
    """
    f = np.linspace(0, fs, N_fft + 1)
    f = f[: int(N_fft / 2) + 1]
    k = np.array(range(numBands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * mn
    freq_low = mn * np.power(2.0, (2 * k - 1) / 6)
    freq_high = mn * np.power(2.0, (2 * k + 1) / 6)
    A = np.zeros((numBands, len(f)))

    for i in range(len(cf)):
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        A[i, fl_ii:fh_ii] = 1
    return A


def removeSilentFrames(x, y, dyn_range=40, N=256, K=128):
    w = np.hanning(256)
    frames = range(0, x.shape[0] - N, K)
    energy = []
    for frame in frames:
        energy.append(
            20
            * np.log10(
                np.linalg.norm(w * x[frame : (frame + N)]) / 16.0 + smallVal
            )
        )

    msk = np.zeros(len(frames))
    Max_energy = max(energy)
    i = 0
    for e in energy:
        if e - Max_energy + dyn_range > 0:
            msk[i] = 1
        i = i + 1

    x_sil = np.zeros(x.shape)
    y_sil = np.zeros(y.shape)
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


def stoi(y_true, y_pred, fs):
    octave_band = thirdoct(fs=10000, N_fft=512, numBands=15, mn=150)

    y_true = resample(y_true, fs, 10000)
    y_pred = resample(y_pred, fs, 10000)

    [y_sil_true, y_sil_pred] = removeSilentFrames(y_true, y_pred)

    _, _, stft_true = signal.stft(
        y_sil_true, fs, nfft=512, nperseg=256, noverlap=128
    )
    _, _, stft_pred = signal.stft(
        y_sil_pred, fs, nfft=512, nperseg=256, noverlap=128
    )

    OCT_true = np.sqrt(np.matmul(octave_band, np.square(np.abs(stft_true))))
    OCT_pred = np.sqrt(np.matmul(octave_band, np.square(np.abs(stft_pred))))

    M = int(
        stft_pred.shape[-1] - (N - 1)
    )  # number of temporal envelope vectors

    d = 0
    for m in range(0, M):  # Run over temporal envelope vectors
        x = OCT_true[:, m : m + N]
        y = OCT_pred[:, m : m + N]

        alpha = np.sqrt(
            np.sum(np.square(x), axis=-1, keepdims=True)
            / (np.sum(np.square(y), axis=-1, keepdims=True) + smallVal)
        )

        ay = y * alpha
        y = np.minimum(ay, x + x * c)

        xn = x - np.mean(x, axis=-1, keepdims=True)
        xn = xn / (np.sqrt(np.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)

        yn = y - np.mean(y, axis=-1, keepdims=True)
        yn = yn / (np.sqrt(np.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)

        di = np.sum(xn * yn, axis=-1, keepdims=True)
        d = d + np.sum(di, axis=0, keepdims=False)

    return d / (J * M)
