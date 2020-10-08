import torch
import torch.nn as nn

import copy


from speechbrain.lobes.models.transformer.TranformerXL import (
    TransformerEncoderRP,
)

from speechbrain.nnet.quantization import quant_noise
from .dual_pathrnn import select_norm

EPS = 1e-8


class SBTransformerBlockRP(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn=2048,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
    ):
        super(SBTransformerBlockRP, self).__init__()

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoderRP(
            num_layers,
            nhead,
            d_ffn,
            kdim,
            vdim,
            dropout,
            activation,
            return_attention,
            num_modules,
            use_group_comm,
        )

    def forward(self, x, pos_embs, init_params=False):

        return self.mdl(
            x.transpose(0, 1), pos_embs, init_params=init_params
        ).transpose(0, 1)


class Dual_Computation_Block(nn.Module):
    """
#            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
    """

    def __init__(self, intra_mdl, inter_mdl):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl

    def forward(self, x, pos_embs_intra, pos_embs_inter, init_params=True):
        """
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(
            intra, pos_embs_intra, init_params=init_params
        ).view(B, S, K, N)
        intra = intra.permute(0, 3, 2, 1).contiguous()
        # [B, N, K, S]
        # out = intra

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter = self.inter_mdl(
            inter, pos_embs_inter, init_params=init_params
        ).view(B, K, S, N)
        inter = inter.permute(0, 3, 1, 2).contiguous()
        out = inter
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, positions: torch.LongTensor,  # (seq, )
    ):
        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        position="relative",
        qnoise_p=0.0,
        qnoise_block=1,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.qnoise_p = qnoise_p
        self.qnoise_block = qnoise_block
        self.position = position

        assert position in ["relative"]

        self.pos_emb = PositionalEmbedding(out_channels)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(Dual_Computation_Block(intra_model, inter_model))
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def apply_quantization_module(self):
        # NOTE APPLY AFTER INIT_PARAMS HAS BEEN CALLED !!!
        modules = list(self.modules())
        for i in range(len(modules)):
            if isinstance(
                modules[i], (nn.Conv2d, nn.Linear, nn.Embedding, nn.Conv1d)
            ):
                modules[i] = quant_noise(
                    modules[i], self.qnoise_p, self.qnoise_block
                )

    def forward(self, x, init_params=True):
        """
           x: [B, N, L]
        """

        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        if self.position == "relative":
            # inter positional embs
            pos_intra = self.pos_emb(
                torch.arange(0, self.K, dtype=torch.float).to(x.device)
            )
            pos_inter = self.pos_emb(
                torch.arange(0, x.size(-1), dtype=torch.float).to(x.device)
            )
        else:
            raise NotImplementedError

        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](
                x, pos_intra, pos_inter, init_params=init_params
            )

        # self.dual_mdl[1].inter_mdl.mdl.layers[0].linear1.weight to see the weights

        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        if init_params:
            self.apply_quantization_module()

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]
        return input
