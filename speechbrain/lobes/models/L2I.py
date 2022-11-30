import torch.nn as nn
import torch.nn.functional as F
import torch


class Psi(nn.Module):
    """ Convolutional Layers to estimate NMF Activations from Classifier Representations

    Arguments
    ---------
    n_comp : int
        Number of NMF components (or equivalently number of neurons at the output per timestep)
    T: int
        The targeted length along the time dimension
    in_emb_dims: List with int elements
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
        """
        Computes NMF activations given classifier hidden representations
        """
        super(Psi, self).__init__()
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
        Argument:
            inp: A length 3 list of classifier input representions.
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


class NMFDecoder(nn.Module):
    """This class implements an NMF decoder

    Arguments
    ---------
    n_comp : int
        Number of NMF components
    n_freq : int
        The number of frequency bins in the NMF dictionary
    init_file : str
        The path for the NMF dictionary to be loaded
    device : str
        The device to run the model

    Example:
    --------
    >>> NMF_dec = NMFDecoder(20, 210, device='cpu')
    >>> H = torch.rand(3, 20, 150)
    >>> Xhat = NMF_dec.forward(H)
    >>> print(Xhat.shape)
    torch.Size([3, 210, 150])
    """

    def __init__(self, n_comp=100, n_freq=513, init_file=None, device="cuda"):
        super(NMFDecoder, self).__init__()

        self.W = nn.Parameter(
            0.1 * torch.rand(n_freq, n_comp), requires_grad=True
        ).to(device)
        self.activ = nn.ReLU()

        # we need to fix this! we should download from HF
        if init_file is not None:
            self.W.data = torch.load(
                init_file, map_location=torch.device(device)
            )

    def forward(self, H):
        """The forward pass for NMF given the activations H

        Arguments:
        ---------
        H : torch.Tensor
            The activations Tensor with shape B x n_comp x T

        where B = Batchsize
              n_comp = number of NMF components
              T = number of timepoints
        """
        W = self.activ(self.W)
        W = nn.functional.normalize(W, dim=0, p=2)
        output = torch.matmul(W.unsqueeze(0), H)
        return output

    def return_W(self):
        W = self.W
        W = nn.functional.normalize(self.activ(W), dim=0, p=2)
        return W


class Theta(nn.Module):
    def __init__(self, N_COMP=100, T=431, num_classes=50) -> None:
        super().__init__()
        self.hard_att = nn.Linear(
            T, 1
        )  # collapse time axis using "attention" based pooling
        self.classifier = nn.Sequential(
            nn.Linear(N_COMP, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, psi_out):
        """psi_out is of shape n_batch x n_comp x T
        collapse time axis using "attention" based pooling"""
        theta_out = self.hard_att(psi_out).squeeze(2)
        # print(theta_out.shape)
        # input()
        theta_out = self.classifier(theta_out)
        # print(theta_out.shape)
        # input()
        return theta_out


class NMFEncoder(nn.Module):
    def __init__(self, nlayers) -> None:
        super().__init__()
        self.convenc = nn.Sequential(
            nn.Conv1d(513, 256, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(128, 100, kernel_size=8, padding="same"),
            nn.ReLU(),
        )

    def forward(self, inp):
        """psi_out is of shape n_batch x n_comp x T
        collapse time axis using "attention" based pooling"""
        return self.convenc(inp)
