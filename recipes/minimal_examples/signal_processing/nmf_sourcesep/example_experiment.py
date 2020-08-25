#!/usr/bin/python
import speechbrain as sb
import torch
import speechbrain.processing.NMF as sb_nmf
from speechbrain.processing.features import spectral_magnitude
from speechbrain.data_io.data_io import write_wav_soundfile
import os
import shutil

experiment_dir = os.path.dirname(os.path.realpath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
data_folder = "../../../../samples/audio_samples/sourcesep_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(hyperparams_file) as fin:
    hyperparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

sb.core.create_experiment_directory(
    experiment_directory=hyperparams.output_folder,
    hyperparams_to_save=hyperparams_file,
)
torch.manual_seed(0)


class NMF_Brain(sb.core.Brain):
    """
    This class implements Non-Negative Matrix Factorization for source
    separation as described in
    https://web.stanford.edu/class/stats253/IEEE_SPM.pdf

    Note that this method does not utilize SGD, but rather multiplicative
    update rules to optimize the model parameters as described in the
    article above.

    Author: Cem Subakan; Mila, Quebec AI Institute
    """

    def __init__(self, loader):
        # over riding the init of Brain class, as we don't deal with neural nets in NMF.
        self.init_matrices(loader)
        self.modules = torch.nn.ModuleList([])
        self.avg_train_loss = 0.0

    def init_matrices(self, train_loader):
        """
        This function is used to initialize the parameter matrices
        """

        X = list(train_loader)[0]
        X = hyperparams.compute_features(X[0][1])
        X = spectral_magnitude(X, power=2)
        n = X.shape[0] * X.shape[1]

        # initialize
        eps = 1e-20
        w = 0.1 * torch.rand(hyperparams.m, hyperparams.K) + 1
        self.w = w / torch.sum(w, dim=0) + eps

        h = 0.1 * torch.rand(hyperparams.K, n) + 1
        self.h = h / torch.sum(h, dim=0) + eps

    def compute_forward(self, X, init_params=False):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        init_params : bool
            Whether this pass should initialize parameters rather
            than return the results of the forward pass.
        """

        X = hyperparams.compute_features(X[0][1])
        X = spectral_magnitude(X, power=2)

        # concatenate all the inputs
        X = X.reshape(-1, X.size(-1)).t()

        eps = 1e-20
        g = X.sum(dim=0) + eps
        z = X / g

        v = z / (torch.matmul(self.w, self.h) + eps)

        nw = self.w * torch.matmul(v, self.h.t())
        self.w = nw / (torch.sum(nw, dim=0) + eps)

        nh = self.h * torch.matmul(self.w.t(), v)
        # sparsity
        nh = nh + 0.02 * nh ** (1.0 + 0.1)

        self.h = nh / (torch.sum(nh, dim=0) + eps)

        self.h *= g

        deviation = (X - torch.matmul(self.w, self.h)).abs().mean().item()

        return torch.matmul(self.w, self.h), self.w, self.h / g, deviation

    def fit_batch(self, batch):
        inputs = batch
        predictions = self.compute_forward(inputs)
        self.training_out = predictions
        return {"loss": torch.tensor(predictions[-1])}

    def evaluate_batch(self, batch):
        inputs, targets = batch
        output = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(output, targets, train=False)
        stats["loss"] = loss.detach()
        return stats

    def summarize(self, stats, write=False):
        return {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

    def on_epoch_end(self, *args):
        print("The loss is {}".format(args[1]["loss"]))


NMF1 = NMF_Brain(hyperparams.train_loader1())

print("fitting model 1")
NMF1.fit(
    train_set=hyperparams.train_loader1(),
    valid_set=None,
    epoch_counter=range(hyperparams.N_epochs),
    progressbar=False,
)
W1hat = NMF1.training_out[1]

NMF2 = NMF_Brain(hyperparams.train_loader2())

print("fitting model 2")
NMF2.fit(
    train_set=hyperparams.train_loader2(),
    valid_set=None,
    epoch_counter=range(hyperparams.N_epochs),
    progressbar=False,
)
W2hat = NMF2.training_out[1]

# separate
mixture_loader = hyperparams.test_loader()
Xmix = list(mixture_loader)[0]

Xmix = hyperparams.compute_features(Xmix[0][1])
Xmix_mag = spectral_magnitude(Xmix, power=2)

X1hat, X2hat = sb_nmf.NMF_separate_spectra([W1hat, W2hat], Xmix_mag)

x1hats, x2hats = sb_nmf.reconstruct_results(
    X1hat,
    X2hat,
    Xmix.permute(0, 2, 1, 3),
    hyperparams.sample_rate,
    hyperparams.win_length,
    hyperparams.hop_length,
)

if hyperparams.save_reconstructed:
    savepath = "results/save/"
    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    for i, (x1hat, x2hat) in enumerate(zip(x1hats, x2hats)):
        write_wav_soundfile(
            x1hat,
            os.path.join(savepath, "separated_source1_{}.wav".format(i)),
            16000,
        )
        write_wav_soundfile(
            x2hat,
            os.path.join(savepath, "separated_source2_{}.wav".format(i)),
            16000,
        )

    if hyperparams.copy_original_files:
        datapath = "samples/audio_samples/sourcesep_samples"

    filedir = os.path.dirname(os.path.realpath(__file__))
    speechbrain_path = os.path.abspath(os.path.join(filedir, "../../../.."))
    copypath = os.path.realpath(os.path.join(speechbrain_path, datapath))

    all_files = os.listdir(copypath)
    wav_files = [fl for fl in all_files if ".wav" in fl]

    for wav_file in wav_files:
        shutil.copy(copypath + "/" + wav_file, savepath)


def test_NMF():
    # Fill in some check here, comparing x1 and x2 to some expected result
    pass
