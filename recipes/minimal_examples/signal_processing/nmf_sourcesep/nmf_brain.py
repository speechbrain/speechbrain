#!/usr/bin/python
import torch
import speechbrain as sb
from speechbrain.processing.features import spectral_magnitude


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

    def __init__(self, loader, hyperparams):
        # over riding the init of Brain class, as we don't deal with neural nets in NMF.
        self.hyperparams = hyperparams
        self.init_matrices(loader)
        self.modules = torch.nn.ModuleList([])
        self.avg_train_loss = 0.0

    def init_matrices(self, train_loader):
        """
        This function is used to initialize the parameter matrices
        """

        X = list(train_loader)[0]
        X = self.hyperparams.compute_features(X[0][1])
        X = spectral_magnitude(X, power=2)
        n = X.shape[0] * X.shape[1]

        # initialize
        eps = 1e-20
        w = 0.1 * torch.rand(self.hyperparams.m, self.hyperparams.K) + 1
        self.w = w / torch.sum(w, dim=0) + eps

        h = 0.1 * torch.rand(self.hyperparams.K, n) + 1
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

        X = self.hyperparams.compute_features(X[0][1])
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
