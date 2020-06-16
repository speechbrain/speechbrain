#!/usr/bin/python
import speechbrain as sb
import torch
from speechbrain.processing.features import spectral_magnitude
from speechbrain.data_io.data_io import write_wav_soundfile

# import matplotlib.pyplot as plt

import os

params_file = "params.yaml"  # "recipes/minimal_examples/neural_networks/nmf_sourcesep/params.yaml"
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)


class NMF_Brain(sb.core.Brain):
    def __init__(self, loader):
        # over riding the init of Brain class, as we don't deal with neural nets here.
        self.init_matrices(loader)
        self.modules = torch.nn.ModuleList([])
        self.avg_train_loss = 0.0

    def init_matrices(self, train_loader):
        """
        This function is used to initialize the parameter matrices
        """

        # ideally I shouldn't be doing this.
        # Is it possible to directly fetch spectrogram columns to the loader?
        X = list(train_loader)[0]
        X = params.compute_features(X[0][1])
        X = spectral_magnitude(X, power=2)
        n = X.shape[0] * X.shape[1]

        # initialize
        eps = 1e-20
        w = torch.rand(params.m, params.K) + 10
        self.w = w / torch.sum(w, dim=0) + eps

        h = torch.rand(params.K, n) + 10
        self.h = h / torch.sum(h, dim=0) + eps

    def forward(self, X, init_params=False):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        init_params : bool
            Whether this pass should initialize parameters rather
            than return the results of the forward pass.
        """

        # ideally I wouldn't want to be doing this.
        # instead, would it be possible directly fetch spectra in the data loader?

        X = params.compute_features(X[0][1])
        X = spectral_magnitude(X, power=2)

        # concatenate all the inputs
        # X = X.permute(0, 2, 1)
        X = X.reshape(-1, X.size(-1)).t()

        eps = 1e-20
        g = X.sum(dim=0) + eps
        z = X / g

        v = z / (torch.matmul(self.w, self.h) + eps)

        nw = self.w * torch.matmul(v, self.h.t())
        self.w = nw / (torch.sum(nw, dim=0) + eps)

        nh = self.h * torch.matmul(self.w.t(), v)
        self.h = nh / (torch.sum(nh, dim=0) + eps)

        self.h *= g

        deviation = (X - torch.matmul(self.w, self.h)).abs().mean().item()

        return torch.matmul(self.w, self.h), self.w, self.h / g, deviation

    def fit_batch(self, batch):
        inputs = batch
        predictions = self.forward(inputs)
        self.training_out = predictions
        return {"loss": torch.tensor(predictions[-1])}

    def evaluate_batch(self, batch):
        inputs, targets = batch
        output = self.forward(inputs)
        loss, stats = self.compute_objectives(output, targets, train=False)
        stats["loss"] = loss.detach()
        return stats

    def summarize(self, stats, write=False):
        return {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

    def on_epoch_end(self, *args):
        print("The loss is {}".format(args[1]["loss"]))


def separate(Whats, mixture_loader):

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


# def visualize_results(mixture_loader, X1hat, X2hat):
#
#    X = list(mixture_loader)[0]
#
#    X = params.compute_features(X[0][1])
#    X = spectral_magnitude(X, power=2)
#
#    X = X.reshape(-1, X.size(-1)).t().numpy()
#    power = 0.5
#
#    plt.subplot(311)
#    plt.imshow(X ** power)
#    plt.gca().invert_yaxis()
#    plt.title("mixture")
#
#    plt.subplot(312)
#    X1hat = X1hat.permute(1, 0, 2).reshape(params.m, -1).numpy()
#    plt.imshow(X1hat ** power)
#    plt.gca().invert_yaxis()
#    plt.title("estimated source1")
#
#    plt.subplot(313)
#    X2hat = X2hat.permute(1, 0, 2).reshape(params.m, -1).numpy()
#    plt.imshow(X2hat ** power)
#    plt.gca().invert_yaxis()
#    plt.title("estimated source2")


def spectral_phase(stft, power=2, log=False):
    """Returns the phase of a complex spectrogram.

    Arguments
    ---------
    stft : torch.Tensor
        A tensor, output from the stft function.

    Example
    -------

    """
    phase = torch.atan2(stft[:, :, :, 1], stft[:, :, :, 0])

    return phase


def reconstruct_results(mixture_loader, Xhat1, Xhat2):

    savepath = "results/"
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


NMF1 = NMF_Brain(params.train_loader1())

print("fitting model 1")
NMF1.fit(
    train_set=params.train_loader1(),
    valid_set=None,
    epoch_counter=range(params.N_epochs),
)
W1hat = NMF1.training_out[1]

NMF2 = NMF_Brain(params.train_loader2())

print("fitting model 2")
NMF2.fit(
    train_set=params.train_loader2(),
    valid_set=None,
    epoch_counter=range(params.N_epochs),
)
W2hat = NMF2.training_out[1]

mixture_loader = params.test_loader()
X1hat, X2hat = separate([W1hat, W2hat], mixture_loader)

# visualize_results(mixture_loader, X1hat, X2hat)
reconstruct_results(mixture_loader, X1hat, X2hat)

# plt.show()
