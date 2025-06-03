"""This file implements the necessary classes and functions to implement Listen-to-Interpret (L2I) interpretation method from https://arxiv.org/abs/2202.11479v2

 Authors
 * Cem Subakan 2022
 * Francesco Paissan 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.PIQ import ResBlockAudio


class Psi(nn.Module):
    """Convolutional Layers to estimate NMF Activations from Classifier Representations

    Arguments
    ---------
    n_comp : int
        Number of NMF components (or equivalently number of neurons at the output per timestep)
    T : int
        The targeted length along the time dimension
    in_emb_dims : List with int elements
        A list with length 3 that contains the dimensionality of the input dimensions
        The list needs to match the number of channels in the input classifier representations
        The last entry should be the smallest entry

    Example
    -------
    >>> inp = [torch.ones(2, 150, 6, 2), torch.ones(2, 100, 6, 2), torch.ones(2, 50, 12, 5)]
    >>> psi = Psi(n_comp=100, T=120, in_emb_dims=[150, 100, 50])
    >>> h = psi(inp)
    >>> print(h.shape)
    torch.Size([2, 100, 120])
    """

    def __init__(self, n_comp=100, T=431, in_emb_dims=[2048, 1024, 512]):
        super().__init__()
        self.in_emb_dims = in_emb_dims
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.upsamp_time = nn.UpsamplingBilinear2d(size=(T, 1))
        out_c = min(in_emb_dims)

        self.c1 = nn.Conv2d(
            in_emb_dims[0], out_c, kernel_size=3, padding="same"
        )
        self.c2 = nn.Conv2d(
            in_emb_dims[1], out_c, kernel_size=3, padding="same"
        )

        self.out_conv = nn.Conv2d(out_c, n_comp, kernel_size=3, padding="same")

        self.conv = nn.Sequential(
            nn.Conv2d(out_c * 3, out_c, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.act = nn.ReLU()

    def forward(self, inp):
        """This forward function returns the NMF time activations given classifier activations

        Arguments
        ---------
        inp: list
            A length 3 list of classifier input representations.

        Returns
        -------
        NMF time activations
        """
        error = "in PSI doesn't match. The embedding dimensions need to be consistent with the list self.in_emb_dims"
        for i, in_emb_dim in enumerate(self.in_emb_dims):
            # sanity check on shapes
            assert inp[i].shape[1] == self.in_emb_dims[i], (
                "Nr. of channels " + error
            )

        assert inp[0].shape[2] == inp[1].shape[2], "Spatial dimension " + error
        assert inp[0].shape[3] == inp[1].shape[3], "Spatial dimension " + error
        assert 2 * inp[0].shape[3] == (inp[2].shape[3] - 1), (
            "Spatial dimension "
            + error
            + f" 1st (idx 0) element has shape {inp[0].shape[3]} second element (idx 1) has shape {inp[2].shape[3]}"
        )

        x1, x2, x3 = inp

        # upsample inp[0] and inp[1] time and frequency axis once
        x1 = self.upsamp(x1)
        x2 = self.upsamp(x2)

        # compress feature number to the min among given hidden representations
        x1 = self.act(self.c1(x1))
        x2 = self.act(self.c2(x2))

        # for compatibility with cnn14 fixed frequency dimension
        x1 = F.pad(x1, (0, 1, 0, 0))
        x2 = F.pad(x2, (0, 1, 0, 0))
        x = torch.cat((x1, x2, x3), axis=1)

        # upsample time axis and collapse freq
        x = self.upsamp_time(x)

        # mix contribution for the three hidden layers -- work on this when fixing training
        x = self.conv(x)
        x = self.act(self.out_conv(x)).squeeze(3)
        return x


class NMFDecoderAudio(nn.Module):
    """This class implements an NMF decoder

    Arguments
    ---------
    n_comp : int
        Number of NMF components
    n_freq : int
        The number of frequency bins in the NMF dictionary
    device : str
        The device to run the model

    Example
    -------
    >>> NMF_dec = NMFDecoderAudio(20, 210, device='cpu')
    >>> H = torch.rand(1, 20, 150)
    >>> Xhat = NMF_dec.forward(H)
    >>> print(Xhat.shape)
    torch.Size([1, 210, 150])
    """

    def __init__(self, n_comp=100, n_freq=513, device="cuda"):
        super().__init__()

        self.W = nn.Parameter(
            0.1 * torch.rand(n_freq, n_comp), requires_grad=True
        )
        self.activ = nn.ReLU()

    def forward(self, H):
        """The forward pass for NMF given the activations H

        Arguments
        ---------
        H : torch.Tensor
            The activations Tensor with shape B x n_comp x T
            where B = Batchsize
                  n_comp = number of NMF components
                  T = number of timepoints

        Returns
        -------
        output : torch.Tensor
            The NMF outputs
        """
        # Assume input of shape n_batch x n_comp x T

        H = self.activ(H)
        temp = self.activ(self.W).unsqueeze(0)
        output = torch.einsum("bij, bjk -> bik", temp, H)

        return output

    def return_W(self):
        """This function returns the NMF dictionary"""
        W = self.W
        return self.activ(W)


def weights_init(m):
    """
    Applies Xavier initialization to network weights.

    Arguments
    ---------
    m : nn.Module
        Module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class PsiOptimized(nn.Module):
    """Convolutional Layers to estimate NMF Activations from Classifier Representations, optimized for log-spectra.

    Arguments
    ---------
    dim : int
        Dimension of the hidden representations (input to the classifier).
    K : int
        Number of NMF components (or equivalently number of neurons at the output per timestep)
    numclasses : int
        Number of possible classes.
    use_adapter : bool
        `True` if you wish to learn an adapter for the latent representations.
    adapter_reduce_dim: bool
        `True` if the adapter should compress the latent representations.

    Example
    -------
    >>> inp = torch.randn(1, 256, 26, 32)
    >>> psi = PsiOptimized(dim=256, K=100, use_adapter=False, adapter_reduce_dim=False)
    >>> h, inp_ad= psi(inp)
    >>> print(h.shape, inp_ad.shape)
    torch.Size([1, 1, 417, 100]) torch.Size([1, 256, 26, 32])
    """

    def __init__(
        self,
        dim=128,
        K=100,
        numclasses=50,
        use_adapter=False,
        adapter_reduce_dim=True,
    ):
        super().__init__()

        self.use_adapter = use_adapter
        self.adapter_reduce_dim = adapter_reduce_dim
        if use_adapter:
            self.adapter = ResBlockAudio(dim)

            if adapter_reduce_dim:
                self.down = nn.Conv2d(dim, dim, 4, (2, 2), 1)
                self.up = nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, (2, 2), 1),
            nn.ReLU(True),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, 1, 12, 1, 1),
            nn.ReLU(),
            nn.Linear(513, K),
            nn.ReLU(),
        )
        self.apply(weights_init)

    def forward(self, hs):
        """
        Computes forward step.

        Arguments
        ---------
        hs : torch.Tensor
            Latent representations (input to the classifier). Expected shape `torch.Size([B, C, H, W])`.

        Returns
        -------
        NMF activations and adapted representations. Shape `torch.Size([B, 1, T, 100])`. : torch.Tensor
        """
        if self.use_adapter:
            hcat = self.adapter(hs)
        else:
            hcat = hs

        if self.adapter_reduce_dim:
            hcat = self.down(hcat)
            z_q_x_st = self.up(hcat)
            out = self.decoder(z_q_x_st)
        else:
            out = self.decoder(hcat)

        return out, hcat


class Theta(nn.Module):
    """This class implements a linear classifier on top of NMF activations

    Arguments
    ---------
    n_comp : int
        Number of NMF components
    T : int
        Number of Timepoints in the NMF activations
    num_classes : int
        Number of classes that the classifier works with

    Example
    -------
    >>> theta = Theta(30, 120, 50)
    >>> H = torch.rand(1, 30, 120)
    >>> c_hat = theta.forward(H)
    >>> print(c_hat.shape)
    torch.Size([1, 50])
    """

    def __init__(self, n_comp=100, T=431, num_classes=50):
        super().__init__()

        # This linear layer collapses the time axis using "attention" based pooling
        self.hard_att = nn.Linear(T, 1, bias=False)

        # The Linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(n_comp, num_classes, bias=False), nn.Softmax(dim=1)
        )

    def forward(self, H):
        """We first collapse the time axis, and then pass through the linear layer

        Arguments
        ---------
        H : torch.Tensor
            The activations Tensor with shape B x n_comp x T
            where B = Batchsize
                  n_comp = number of NMF components
                  T = number of timepoints

        Returns
        -------
        theta_out : torch.Tensor
            Classifier output
        """
        theta_out = self.hard_att(H).squeeze(2)
        theta_out = self.classifier(theta_out)
        return theta_out


class NMFEncoder(nn.Module):
    """This class implements an NMF encoder with a convolutional network

    Arguments
    ---------
    n_freq : int
        The number of frequency bins in the NMF dictionary
    n_comp : int
        Number of NMF components

    Example
    -------
    >>> nmfencoder = NMFEncoder(513, 100)
    >>> X = torch.rand(1, 513, 240)
    >>> Hhat = nmfencoder(X)
    >>> print(Hhat.shape)
    torch.Size([1, 100, 240])
    """

    def __init__(self, n_freq, n_comp):
        super().__init__()
        self.convenc = nn.Sequential(
            nn.Conv1d(n_freq, 256, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(128, n_comp, kernel_size=8, padding="same"),
            nn.ReLU(),
        )

    def forward(self, X):
        """
        Arguments
        ---------
        X : torch.Tensor
            The input spectrogram Tensor with shape B x n_freq x T
            where B = Batchsize
                  n_freq = nfft for the input spectrogram
                  T = number of timepoints

        Returns
        -------
        NMF encoded outputs.
        """
        return self.convenc(X)


class CNN14PSI_stft(nn.Module):
    """
    This class estimates a saliency map on the STFT domain, given classifier representations.

    Arguments
    ---------
    dim : int
        Dimensionality of the input representations.
    K : int
        Defines the number of output channels in the saliency map.

    Example
    -------
    >>> from speechbrain.lobes.models.Cnn14 import Cnn14
    >>> classifier_embedder = Cnn14(mel_bins=80, emb_dim=2048, return_reps=True)
    >>> x = torch.randn(2, 201, 80)
    >>> _, hs = classifier_embedder(x)
    >>> psimodel = CNN14PSI_stft(2048, 20)
    >>> xhat = psimodel.forward(hs)
    >>> print(xhat.shape)
    torch.Size([2, 20, 207])
    """

    def __init__(self, dim=128, K=100):
        super().__init__()

        self.convt1 = nn.ConvTranspose1d(dim, dim, 3, 2, 1)
        self.convt2 = nn.ConvTranspose1d(dim // 2, dim, 3, 2, 1)
        self.convt3 = nn.ConvTranspose1d(dim, dim, 7, 2, 1)
        self.convt4 = nn.ConvTranspose1d(dim // 4, dim, 5, 2, 1)
        self.convt5 = nn.ConvTranspose1d(dim, dim // 2, 3, 2, 1)
        self.convt6 = nn.ConvTranspose1d(dim // 8, dim // 2, 3, 2, 1)
        self.convt7 = nn.ConvTranspose1d(dim // 2, dim // 4, 4, 2, 0)
        self.convt8 = nn.ConvTranspose1d(dim // 4, dim // 8, 3, 2, 0)
        self.convt9 = nn.ConvTranspose1d(dim // 8, K, 7, 1, 0)

        self.nonl = nn.ReLU(True)

    def forward(self, hs, labels=None):
        """
        Forward step. Estimates NMF activations to be used to get the saliency mask.

        Arguments
        --------
        hs : torch.Tensor
            Classifier's representations.
        labels : torch.Tensor
            Predicted labels for classifier's representations.

        Returns
        --------
        xhat : torch.Tensor
            The estimated NMF activation coefficients
        """

        hs = [h.mean(-1) for h in hs]
        h1 = self.convt1(hs[0])
        h1 = self.nonl(h1)

        h2 = self.convt2(hs[1])
        h2 = self.nonl(h2)
        h = h1 + h2

        h3 = self.convt3(h)
        h3 = self.nonl(h3)

        h4 = self.convt4(hs[2])
        h4 = self.nonl(h4)
        h = h3 + h4

        h5 = self.convt5(h)
        h5 = self.nonl(h5)

        h6 = self.convt6(hs[3])
        h6 = self.nonl(h6)

        h = h5 + h6

        h = self.convt7(h)
        h = self.nonl(h)

        h = self.convt8(h)
        h = self.nonl(h)

        xhat = self.convt9(h)
        xhat = self.nonl(xhat)

        # apply ReLU
        xhat = F.relu(xhat)
        return xhat


class CNN14PSI_stft_2d(nn.Module):
    """
    This class estimates the NMF activations to create a saliency map using the L2I framework

    Arguments
    ---------
    dim : int
        Dimensionality of the input representations.
    K : int
        Defines the number of output channels in the saliency map.

    Example
    -------
    >>> from speechbrain.lobes.models.Cnn14 import Cnn14
    >>> classifier_embedder = Cnn14(mel_bins=80, emb_dim=2048, return_reps=True)
    >>> x = torch.randn(2, 201, 80)
    >>> _, hs = classifier_embedder(x)
    >>> psimodel = CNN14PSI_stft_2d(2048, 20)
    >>> xhat = psimodel.forward(hs)
    >>> print(xhat.shape)
    torch.Size([2, 20, 207])
    """

    def __init__(self, dim=128, K=100):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(dim, dim, 3, (2, 4), 1)
        self.convt2 = nn.ConvTranspose2d(dim // 2, dim, 3, (2, 4), 1)
        self.convt3 = nn.ConvTranspose2d(dim, dim, (7, 4), (2, 4), 1)
        self.convt4 = nn.ConvTranspose2d(dim // 4, dim, (5, 4), (2, 4), 1)
        self.convt5 = nn.ConvTranspose2d(dim, dim // 2, (3, 5), (2, 2), 1)
        self.convt6 = nn.ConvTranspose2d(dim // 8, dim // 2, (3, 3), (2, 4), 1)
        self.convt7 = nn.ConvTranspose2d(
            dim // 2, dim // 4, (4, 3), (2, 2), (0, 5)
        )
        self.convt8 = nn.ConvTranspose2d(
            dim // 4, dim // 8, (3, 4), (2, 2), (0, 2)
        )
        self.convt9 = nn.ConvTranspose2d(dim // 8, K, (7, 5), (1, 4), 0)

        self.nonl = nn.ReLU(True)

    def forward(self, hs, labels=None):
        """
        Forward step. Estimates NMF activations to be used to get the saliency mask.

        Arguments
        --------
        hs : torch.Tensor
            Classifier's representations.
        labels : torch.Tensor
            Predicted labels for classifier's representations.

        Returns
        --------
        xhat : torch.Tensor
            The estimated NMF activation coefficients
        """

        h1 = self.convt1(hs[0])
        h1 = self.nonl(h1)
        # h1 = self.bn1(h1)

        h2 = self.convt2(hs[1])
        h2 = self.nonl(h2)
        # h2 = self.bn2(h2)
        h = h1 + h2

        h3 = self.convt3(h)
        h3 = self.nonl(h3)
        # h3 = self.bn3(h3)

        h4 = self.convt4(hs[2])
        h4 = self.nonl(h4)
        # h4 = self.bn4(h4)
        h = h3 + h4

        h5 = self.convt5(h)
        h5 = self.nonl(h5)
        # h5 = self.bn5(h5)

        h6 = self.convt6(hs[3])
        h6 = self.nonl(h6)
        # h6 = self.bn6(h6)

        h = h5 + h6

        h = self.convt7(h)
        h = self.nonl(h)
        # h = self.bn7(h)

        h = self.convt8(h)
        h = self.nonl(h)

        xhat = self.convt9(h)
        xhat = self.nonl(xhat)

        xhat = xhat.mean(-1)

        # apply ReLU
        xhat = F.relu(xhat)
        return xhat
