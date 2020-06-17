import os
import torch
from speechbrain.data_io.data_io import write_wav_soundfile
from speechbrain.processing.features import spectral_magnitude

# import matplotlib.pyplot as plt


def spectral_phase(stft, power=2, log=False):
    """Returns the phase of a complex spectrogram.

    Arguments
    ---------
    stft : torch.Tensor
        A tensor, output from the stft function.

    Example
    -------
    phase_mix = spectral_phase(X_stft).permute(0, 2, 1)

    """
    phase = torch.atan2(stft[:, :, :, 1], stft[:, :, :, 0])

    return phase


def separate(params, Whats, mixture_loader):
    """This function separates the mixture signals, given NMF template matrices

    Arguments
    ---------
    params : dict
        This is the experiment dictionary that comes from the experiment
        yaml file.
    Whats : list
        This list contains the list [W1, W2], where W1 W2 are respectively
        the NMF template matrices that correspond to source1 and source2.
    mixture_loader : data_loader
        This loader contains the mixture signals to be separated.

    Example
    -------
    >>> X1hat, X2hat = separate(params, [W1hat, W2hat], mixture_loader)
    """

    W1, W2 = Whats

    X = list(mixture_loader)[0]

    X = params.compute_features(X[0][1])
    X = spectral_magnitude(X, power=2)

    # concatenate all the inputs
    # X = X.permute(0, 2, 1)

    X = X.reshape(-1, X.size(-1)).t()

    n = X.shape[1]
    eps = 1e-20

    # Normalize input
    g = X.sum(dim=0) + eps
    z = X / g

    # initialize
    w = torch.cat([W1, W2], dim=1)
    K = w.size(1)
    K1 = W1.size(1)
    K2 = W2.size(1)

    h = torch.rand(K, n) + 10
    h /= torch.sum(h, dim=0) + eps

    for ep in range(200):
        v = z / (torch.matmul(w, h) + eps)

        nh = h * torch.matmul(w.t(), v)
        h = nh / (torch.sum(nh, dim=0) + eps)

    h *= g
    Xhat1 = torch.matmul(w[:, :K1], h[:K1, :])
    Xhat1 = torch.split(Xhat1.unsqueeze(0), Xhat1.size(1) // 2, dim=2)
    Xhat1 = torch.cat(Xhat1, dim=0)

    Xhat2 = torch.matmul(w[:, K2:], h[K2:, :])
    Xhat2 = torch.split(Xhat2.unsqueeze(0), Xhat2.size(1) // 2, dim=2)
    Xhat2 = torch.cat(Xhat2, dim=0)

    return Xhat1, Xhat2


def reconstruct_results(params, mixture_loader, Xhat1, Xhat2):

    """This function reconstructs the separated spectra into waveforms.

    Arguments
    ---------
    params : dict
        This is the experiment dictionary that comes from the experiment
        yaml file.
    mixture_loader : data_loader
        This loader contains the mixture signals to be separated.
    Xhat1 : torch_tensor
        The separated spectrum for source 1 of size [BS, nfft/2 + 1, T].
    Xhat2 : torch_tensor
        The separated spectrum for source 2 of size [BS, nfft/2 + 1, T].


    Example
    -------
    >>> reconstruct_results(params, mixture_loader, X1hat, X2hat)
    """

    savepath = "output_folder/save/"
    if not os.path.exists("output_folder"):
        os.mkdir("output_folder")

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    X = list(mixture_loader)[0]

    X_stft = params.compute_features(X[0][1])
    phase_mix = spectral_phase(X_stft).permute(0, 2, 1)
    mag_mix = spectral_magnitude(X_stft, power=2).permute(0, 2, 1)

    eps = 1e-25
    for i in range(Xhat1.shape[0]):
        Xhat1_stft = (
            (Xhat1[i] / (eps + Xhat1[i] + Xhat2[i])).unsqueeze(-1)
            * mag_mix[i].unsqueeze(-1)
            * torch.cat(
                [
                    torch.cos(phase_mix[i].unsqueeze(-1)),
                    torch.sin(phase_mix[i].unsqueeze(-1)),
                ],
                dim=-1,
            )
        )

        Xhat2_stft = (
            (Xhat2[i] / (eps + Xhat1[i] + Xhat2[i])).unsqueeze(-1)
            * mag_mix[i].unsqueeze(-1)
            * torch.cat(
                [
                    torch.cos(phase_mix[i].unsqueeze(-1)),
                    torch.sin(phase_mix[i].unsqueeze(-1)),
                ],
                dim=-1,
            )
        )

        shat1 = params.istft(Xhat1_stft.unsqueeze(0).permute(0, 2, 1, 3))

        shat2 = params.istft(Xhat2_stft.unsqueeze(0).permute(0, 2, 1, 3))

        write_wav_soundfile(
            shat1 / (3 * shat1.std()),
            savepath + "s1hat_{}".format(i) + ".wav",
            16000,
        )
        write_wav_soundfile(
            shat2 / (3 * shat2.std()),
            savepath + "s2hat_{}".format(i) + ".wav",
            16000,
        )
