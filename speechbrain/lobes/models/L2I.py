import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import speechbrain as sb


class Psi(nn.Module):
    def __init__(self, N_COMP=100, T=431, in_maps=[2048, 1024, 512]):
        """
        Computes NMF dictionary activations given classifier hidden layers
        """
        super(Psi, self).__init__()
        self.in_maps = in_maps
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.upsamp_time = nn.UpsamplingBilinear2d(size=(T, 1))
        out_c = min(in_maps)

        self.c1 = nn.Conv2d(in_maps[0], out_c, kernel_size=3, padding="same")
        self.c2 = nn.Conv2d(in_maps[1], out_c, kernel_size=3, padding="same")

        self.out_conv = nn.Conv2d(out_c, N_COMP, kernel_size=3, padding="same")

        self.conv = nn.Sequential(
            nn.Conv2d(out_c * 3, out_c, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.act = nn.ReLU()

    def forward(self, inp):
        """
        `inp` contains the hidden representations from the network.
        inp[0] and inp[1] need a factor 2 upsampling on the time axis, while inp[0] just needs features to match K
        """
        error = "in PSI doesn't match. Did you change the classifier model?"
        for i in range(len(self.in_maps)):
            # sanity check on shapes
            assert inp[0].shape[1] == self.in_maps[0], (
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

        # print(x1.shape, x2.shape, x3.shape)
        # input()

        # upsample inp[0] and inp[1] time and frequency axis once
        x1 = self.upsamp(x1)
        x2 = self.upsamp(x2)

        # compress feature number to the min among given hidden repr
        x1 = self.act(self.c1(x1))
        x2 = self.act(self.c2(x2))

        # for cnn14 fix frequency dimension
        x1 = F.pad(x1, (0, 1, 0, 0))
        x2 = F.pad(x2, (0, 1, 0, 0))

        # print(x1.shape, x2.shape, x3.shape)
        # input()

        x = torch.cat((x1, x2, x3), axis=1)

        # upsample time axis and collapse freq
        x = self.upsamp_time(x)

        # mix contribution for the three hidden layers -- work on this when fixing training
        x = self.conv(x)
        x = self.act(self.out_conv(x)).squeeze(3)

        return x


class Psi_independent(nn.Module):
    def __init__(self, N_COMP=100, T=431, in_maps=[2048, 1024, 512]):
        """
        Computes NMF dictionary activations given classifier hidden layers
        """
        super(Psi_independent, self).__init__()

        self.T = T
        self.K = N_COMP
        self.in_maps = in_maps
        self.h = nn.Parameter(
            data=torch.rand(self.K, self.T, requires_grad=True)
        )

    def forward(self, inp):
        """
        `inp` contains the hidden representations from the network.
        inp[0] and inp[1] need a factor 2 upsampling on the time axis, while inp[0] just needs features to match K
        """
        return F.relu(self.h)


class Psi_ConvTranspose(nn.Module):
    def __init__(self, N_COMP=100, T=431, in_maps=[2048, 1024, 512]):
        """
        Computes NMF dictionary activations given classifier hidden layers
        """
        super(Psi_ConvTranspose, self).__init__()

        self.in_maps = in_maps
        self.c1_1 = sb.nnet.CNN.ConvTranspose1d(
            1024,
            kernel_size=8,
            in_channels=in_maps[0],
            stride=2,
            padding=6,
            skip_transpose=True,
        )
        self.c1_2 = sb.nnet.CNN.ConvTranspose1d(
            512,
            kernel_size=8,
            in_channels=1024,
            stride=2,
            padding=4,
            skip_transpose=True,
        )
        self.c1_3 = sb.nnet.CNN.ConvTranspose1d(
            256,
            kernel_size=8,
            in_channels=512,
            stride=3,
            padding=0,
            skip_transpose=True,
        )
        self.c1_4 = sb.nnet.CNN.ConvTranspose1d(
            128,
            kernel_size=8,
            in_channels=256,
            stride=2,
            padding=0,
            skip_transpose=True,
        )
        self.c1_5 = sb.nnet.CNN.ConvTranspose1d(
            N_COMP,
            kernel_size=8,
            in_channels=128,
            stride=2,
            padding=0,
            skip_transpose=True,
        )

        self.c2_1 = sb.nnet.CNN.ConvTranspose1d(
            1024,
            kernel_size=8,
            in_channels=in_maps[1],
            stride=2,
            padding=6,
            skip_transpose=True,
        )

        self.c3_1 = sb.nnet.CNN.ConvTranspose1d(
            256,
            kernel_size=8,
            in_channels=in_maps[2],
            stride=2,
            padding=6,
            skip_transpose=True,
        )

    def forward(self, inp):
        """
        `inp` contains the hidden representations from the network.
        inp[0] and inp[1] need a factor 2 upsampling on the time axis, while inp[0] just needs features to match K
        """
        error = "in PSI doesn't match. Did you change the classifier model?"
        for i in range(len(self.in_maps)):
            # sanity check on shapes
            assert inp[0].shape[1] == self.in_maps[0], (
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
        # x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)
        x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
        x3 = x3.reshape(x3.shape[0], x3.shape[1], -1)
        x1 = x1.mean(-1)

        x1 = self.c1_1(x1)
        x1 = self.c1_2(x1)
        x1 = self.c1_3(x1)
        x1 = self.c1_4(x1)
        x1 = self.c1_5(x1)

        return F.relu(x1)


class NMFDecoder(nn.Module):
    def __init__(self, N_COMP=100, FREQ=513, init_file=None, device="cuda"):
        super(NMFDecoder, self).__init__()

        self.W = nn.Parameter(
            0.1 * torch.rand(FREQ, N_COMP), requires_grad=True
        )
        self.activ = nn.ReLU()

        if init_file is not None:
            # handle numpy or torch
            if ".pt" in init_file:
                self.W.data = torch.load(init_file, map_location=torch.device(device))
            else:
                temp = np.load(init_file)
                self.W.data = torch.as_tensor(temp).float()

    def forward(self, inp):
        # Assume input of shape n_batch x n_comp x T
        W = self.activ(self.W)
        W = nn.functional.normalize(W, dim=0, p=2)

        W = torch.stack(
            inp.shape[0] * [W], dim=0
        )  # use same NMF dictionary for every element in the batch

        output = self.activ(torch.bmm(W, inp))  # .transpose(1, -1)

        return output

    def return_W(self, dtype="numpy"):
        W = self.W
        W = nn.functional.normalize(self.activ(W), dim=0, p=2)
        if dtype == "numpy":
            return W.cpu().data.numpy()
        else:
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
