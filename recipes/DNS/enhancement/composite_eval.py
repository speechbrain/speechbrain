"""Composite objective enhancement scores in Python (CSIG, CBAK, COVL)

Taken from https://github.com/facebookresearch/denoiser/blob/master/scripts/matlab_eval.py

Authors
 * adiyoss (https://github.com/adiyoss)
"""

from scipy.linalg import toeplitz
from tqdm import tqdm
from pesq import pesq
import librosa
import numpy as np
import os
import sys


def eval_composite(ref_wav, deg_wav, sample_rate):
    """Evaluate audio quality metrics based on reference
    and degraded audio signals.
    This function computes various audio quality metrics,
    including PESQ, CSIG, CBAK, and COVL, based on the
    reference and degraded audio signals provided.
    """
    ref_wav = ref_wav.reshape(-1)
    deg_wav = deg_wav.reshape(-1)

    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    deg_wav = deg_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(ref_wav, deg_wav, sample_rate)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[: int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(ref_wav, deg_wav, sample_rate)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(ref_wav, deg_wav, sample_rate)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = PESQ(ref_wav, deg_wav, sample_rate)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)

    return {"pesq": pesq_raw, "csig": Csig, "cbak": Cbak, "covl": Covl}


# ----------------------------- HELPERS ------------------------------------ #
def trim_mos(val):
    """Trim a value to be within the MOS (Mean Opinion Score)
    range [1, 5].
    """
    return min(max(val, 1), 5)


def lpcoeff(speech_frame, model_order):
    """Calculate linear prediction (LP) coefficients using
    the autocorrelation method.
    """
    # (1) Compute Autocor lags
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first = speech_frame[: (winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))

    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr = np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)

    return acorr, refcoeff, lpparams


# -------------------------------------------------------------------------- #

# ---------------------- Speech Quality Metric ----------------------------- #
def PESQ(ref_wav, deg_wav, sample_rate):
    """Compute PESQ score.
    """
    psq_mode = "wb" if sample_rate == 16000 else "nb"
    return pesq(sample_rate, ref_wav, deg_wav, psq_mode)


def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]

    # scale both to have same dynamic range. Remove DC too.
    clean_speech -= clean_speech.mean()
    processed_speech -= processed_speech.mean()
    processed_speech *= np.max(np.abs(clean_speech)) / np.max(
        np.abs(processed_speech)
    )

    # Signal-to-Noise Ratio
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(
        np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + 10e-20)
    )
    # global variables
    winlength = int(np.round(30 * srate / 1000))  # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(
            10 * np.log10(signal_energy / (noise_energy + eps) + eps)
        )
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr


def wss(ref_wav, deg_wav, srate):
    """ Calculate Weighted Spectral Slope (WSS) distortion
    measure between reference and degraded audio signals.
    This function computes the WSS distortion measure using
    critical band filters and spectral slope differences.
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.0)  # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25  # num of critical bands

    n_fft = int(2 ** np.ceil(np.log(2 * winlength) / np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions (Center frequency and BW in Hz)
    cent_freq = [
        50.0,
        120,
        190,
        260,
        330,
        400,
        470,
        540,
        617.372,
        703.378,
        798.717,
        904.128,
        1020.38,
        1148.30,
        1288.72,
        1442.54,
        1610.70,
        1794.16,
        1993.93,
        2211.08,
        2446.71,
        2701.97,
        2978.04,
        3276.17,
        3597.63,
    ]
    bandwidth = [
        70.0,
        70,
        70,
        70,
        70,
        70,
        70,
        77.3724,
        86.0056,
        95.3398,
        105.411,
        116.256,
        127.914,
        140.423,
        153.823,
        168.154,
        183.457,
        199.776,
        217.153,
        235.631,
        255.255,
        276.072,
        298.126,
        321.465,
        346.136,
    ]

    bw_min = bandwidth[0]  # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.
    min_factor = np.exp(-30.0 / (2 * 2.303))  # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(
            -11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor
        )
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    # For each frame of input speech, compute Weighted Spectral Slope Measure
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0  # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed
        clean_spec = np.abs(np.fft.fft(clean_frame, n_fft)) ** 2
        processed_spec = np.abs(np.fft.fft(processed_frame, n_fft)) ** 2
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit

        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * crit_filter[i, :])
            processed_energy[i] = np.sum(
                processed_spec[:n_fftby2] * crit_filter[i, :]
            )
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))

        # (4) Compute Spectral Shape (dB[i+1] - dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[: num_crit - 1]
        processed_slope = (
            processed_energy[1:num_crit] - processed_energy[: num_crit - 1]
        )

        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])

        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)

        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[: num_crit - 1])
        Wlocmax_clean = Klocmax / (
            Klocmax + clean_loc_peak - clean_energy[: num_crit - 1]
        )
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (
            Kmax + dBMax_processed - processed_energy[: num_crit - 1]
        )
        Wlocmax_processed = Klocmax / (
            Klocmax + processed_loc_peak - processed_energy[: num_crit - 1]
        )
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(
            np.sum(
                W
                * (
                    clean_slope[: num_crit - 1]
                    - processed_slope[: num_crit - 1]
                )
                ** 2
            )
        )

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion


def llr(ref_wav, deg_wav, srate):
    """Calculate Log Likelihood Ratio (LLR) distortion measure
    between reference and degraded audio signals. This function
    computes the LLR distortion measure between reference and
    degraded audio signals using LPC analysis and autocorrelation
    logs.
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.0)  # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]

        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)

        if (numerator / denominator) <= 0:
            print(f"Numerator: {numerator}")
            print(f"Denominator: {denominator}")

        log_ = np.log(numerator / denominator)
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.nan_to_num(np.array(distortion))


# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    clean_path = sys.argv[1]
    enhanced_path = sys.argv[2]
    csig, cbak, covl, count = 0, 0, 0, 0
    for _file in tqdm(os.listdir(clean_path)):
        if _file.endswith("wav"):
            clean_path_f = os.path.join(clean_path, _file)
            enhanced_path_f = os.path.join(
                enhanced_path, _file[:-4] + "_enhanced.wav"
            )
            clean_sig = librosa.load(clean_path_f, sr=None)[0]
            enhanced_sig = librosa.load(enhanced_path_f, sr=None)[0]
            res = eval_composite(clean_sig, enhanced_sig)
            csig += res["csig"]
            cbak += res["cbak"]
            covl += res["covl"]
            pesq += res["pesq"]
            count += 1
    print(f"CSIG: {csig/count}, CBAK: {cbak/count}, COVL: {covl/count}")
