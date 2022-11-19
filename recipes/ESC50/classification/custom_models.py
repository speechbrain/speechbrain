import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import speechbrain as sb


class weak_mxh64_1024(nn.Module):
    def __init__(self, nout):
        super(weak_mxh64_1024, self).__init__()

        glplfn = F.max_pool2d

        self.globalpool = glplfn
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer3 = nn.MaxPool2d(2)

        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer6 = nn.MaxPool2d(2)

        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer9 = nn.MaxPool2d(2)

        self.layer10 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer12 = nn.MaxPool2d(2)

        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer15 = nn.MaxPool2d(2)  #

        self.layer16 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer17 = nn.MaxPool2d(2)  #

        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, nout, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out1 = self.layer19(out)
        out = self.globalpool(out1, kernel_size=out1.size()[2:])
        out = out.view(out.size(0), -1)
        return out  # ,out1


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.norm_type = norm_type

        if norm_type == "bn":
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == "in":
            #  self.norm1 = nn.GroupNorm(out_channels, out_channels)
            #  self.norm2 = nn.GroupNorm(out_channels, out_channels)
            self.norm1 = nn.InstanceNorm2d(
                out_channels, affine=True, track_running_stats=True
            )
            self.norm2 = nn.InstanceNorm2d(
                out_channels, affine=True, track_running_stats=True
            )
        elif norm_type == "ln":
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)
        else:
            raise ValueError("Unknown norm type {}".format(norm_type))

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.norm1)
        init_bn(self.norm2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.norm1(self.conv1(x)))
        x = F.relu_(self.norm2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    #  def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
    #      fmax, classes_num):
    def __init__(self, mel_bins, emb_dim, norm_type="bn", interpret=False):

        super(Cnn14, self).__init__()
        self.interpret = interpret

        #  window = 'hann'
        #  center = True
        #  pad_mode = 'reflect'
        #  ref = 1.0
        #  amin = 1e-10
        #  top_db = None

        #  # Spectrogram extractor
        #  self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #      win_length=window_size, window=window, center=center, pad_mode=pad_mode,
        #      freeze_parameters=True)
        #
        #  # Logmel feature extractor
        #  self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #      n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
        #      freeze_parameters=True)
        #
        #  # Spec augmenter
        #  self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #      freq_drop_width=8, freq_stripes_num=2)
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm0 = nn.BatchNorm2d(mel_bins)
        elif norm_type == "in":
            #  self.norm0 = nn.GroupNorm(mel_bins, mel_bins)
            self.norm0 = nn.InstanceNorm2d(
                mel_bins, affine=True, track_running_stats=True
            )
        elif norm_type == "ln":
            self.norm0 = nn.GroupNorm(1, mel_bins)
        else:
            raise ValueError("Unknown norm type {}".format(norm_type))

        self.conv_block1 = ConvBlock(
            in_channels=1, out_channels=64, norm_type=norm_type
        )
        self.conv_block2 = ConvBlock(
            in_channels=64, out_channels=128, norm_type=norm_type
        )
        self.conv_block3 = ConvBlock(
            in_channels=128, out_channels=256, norm_type=norm_type
        )
        self.conv_block4 = ConvBlock(
            in_channels=256, out_channels=512, norm_type=norm_type
        )
        self.conv_block5 = ConvBlock(
            in_channels=512, out_channels=1024, norm_type=norm_type
        )
        self.conv_block6 = ConvBlock(
            in_channels=1024, out_channels=emb_dim, norm_type=norm_type
        )

        #  self.fc1 = nn.Linear(2048, 2048, bias=True)
        #  self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.norm0)
        #  init_layer(self.fc1)
        #  init_layer(self.fc_audioset)

    def forward(self, x, mixup_lambda=None):
        """
        Input: (B, 1, T, M)"""

        #  x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        #  x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.norm0(x)
        x = x.transpose(1, 3)

        #  if self.training:
        #      x = self.spec_augmenter(x)
        #
        #  # Mixup on spectrogram
        #  if self.training and mixup_lambda is not None:
        #      x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x3_out = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x3_out, p=0.2, training=self.training)
        x2_out = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x2_out, p=0.2, training=self.training)
        x1_out = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x1_out, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        if not self.interpret:
            return x.unsqueeze(1)  # [B, 1, D]

        return x.unsqueeze(1), (x1_out, x2_out, x3_out)

        #  x = F.dropout(x, p=0.5, training=self.training)
        #  x = F.relu_(self.fc1(x))
        #  embedding = F.dropout(x, p=0.5, training=self.training)
        #  clipwise_output = torch.sigmoid(self.fc_audioset(x))
        #
        #  output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        #
        #  return output_dict


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
    def __init__(self, N_COMP=100, FREQ=513, init_file=None):
        super(NMFDecoder, self).__init__()

        self.W = nn.Parameter(0.1 * torch.rand(FREQ, N_COMP), requires_grad=True)
        self.activ = nn.ReLU()

        if init_file is not None:
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
            nn.Conv1d(513, 256, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 100, kernel_size=8, padding='same'),
            nn.ReLU(),
        )

    def forward(self, inp):
        """psi_out is of shape n_batch x n_comp x T
        collapse time axis using "attention" based pooling"""
        return self.convenc(inp)
