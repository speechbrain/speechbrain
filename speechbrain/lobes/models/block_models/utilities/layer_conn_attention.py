from .attention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerConnAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, d_out, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(
            self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k))
        )
        nn.init.normal_(
            self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k))
        )
        nn.init.normal_(
            self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v))
        )

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5)
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.gate_fc = nn.Linear(n_head * d_v, d_out)
        self.fc = nn.Linear(n_head * d_v, d_out)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        # print('attn input shape', q.shape)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # v = v.view(sz_b, len_v, n_head, d_v)

        q = (
            q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        )  # (n*b) x lq x dk
        k = (
            k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        )  # (n*b) x lk x dk
        v = (
            v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        )  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn, extra_loss = self.attention(q, k, v, mask=None)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        # print('output shape before fc', output.shape)

        # TODO: probably shouldn't just apply residual layer in the forward pass.

        output_init = output * 1.0

        # output = self.dropout(self.fc(output_init))
        output = self.dropout(output_init)

        # gate = F.sigmoid(self.gate_fc(output_init))

        # output = self.layer_norm(gate * output + (1 - gate) * residual)
        # output = gate * output + (1 - gate) * residual

        # output = residual + gate * F.tanh(output)

        # output

        # print('attn', attn[0])
        # print('output input diff', output - residual)

        return output, attn, extra_loss
