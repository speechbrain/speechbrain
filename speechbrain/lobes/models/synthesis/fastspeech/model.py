import torch
import math
import numpy as np
from torch import nn
import sys
sys.path.append('../../../')
import torch.nn as nn
from torch.nn import functional as F
from speechbrain.nnet import CNN
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.dropout import Dropout2d
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder, TransformerDecoder, get_key_padding_mask, get_mel_mask


class EncoderPreNet(nn.Module):
    def __init__(self, n_vocab, blank_id, out_channels=512, dropout=0.15, num_layers=3, device='cuda:0'):
        super().__init__()
        self.phoneme_embedding = Embedding(num_embeddings=n_vocab, embedding_dim=out_channels, blank_id=blank_id).to(device)

    def forward(self, x):
        x = self.phoneme_embedding(x)
        return x


# class DurationPredictor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = CNN.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
#         self.conv2 = CNN.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
#         self.linear = Linear(input_size=out_channels, n_neurons=1)
#         self.ln1 = nn.LayerNorm(out_channels)
#         self.ln2 = nn.LayerNorm(out_channels)
#
#     def forward(self, x):
#         x = self.ln1(self.conv1(x))
#         x = self.ln2(self.conv2(x))
#         return self.linear(x)
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.linear = nn.Linear(out_channels, 1)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        x = self.relu(x)
        x = self.ln1(x.transpose(1, 2)).to(x.dtype)

        x = self.conv2(x.transpose(1, 2))
        x = self.relu(x)
        x = self.ln2(x.transpose(1, 2)).to(x.dtype)

        return self.linear(x)

class FastSpeech(nn.Module):
    def __init__(self, pre_net_dropout,
                    pre_net_num_layers,
                    post_net_num_layers,
                    enc_num_layers,
                    enc_num_head,
                    enc_d_model,
                    enc_ffn_dim,
                    enc_k_dim,
                    enc_v_dim,
                    enc_dropout,
                    dec_num_layers,
                    dec_num_head,
                    dec_d_model,
                    dec_ffn_dim,
                    dec_k_dim,
                    dec_v_dim,
                    dec_dropout,
                    normalize_before,
                    ffn_type,
                    n_char,
                    n_mels,
                    padding_idx,
                    dur_pred_kernel_size):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx

        self.encPreNet = EncoderPreNet(n_char, padding_idx, out_channels=enc_d_model)
        self.durPred = DurationPredictor(in_channels=enc_d_model, out_channels=enc_d_model, kernel_size=dur_pred_kernel_size)
        self.encoder = TransformerEncoder(num_layers=enc_num_layers,
                                        nhead=enc_num_head,
                                        d_ffn=enc_ffn_dim,
                                        d_model=enc_d_model,
                                        kdim=enc_k_dim,
                                        vdim=enc_v_dim,
                                        dropout=enc_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before,
                                        ffn_type=ffn_type)

        self.decoder = TransformerEncoder(num_layers=dec_num_layers,
                                        nhead=dec_num_head,
                                        d_ffn=dec_ffn_dim,
                                        d_model=dec_d_model,
                                        kdim=dec_k_dim,
                                        vdim=dec_v_dim,
                                        dropout=dec_dropout,
                                        activation=nn.ReLU,
                                        normalize_before=normalize_before,
                                        ffn_type=ffn_type)

        self.linear = nn.Linear(dec_d_model, n_mels)


    def forward(self, phonemes, durations=None):
        # print(phonemes, durations)
        import matplotlib.pyplot as plt
        phoneme_feats = self.encPreNet(phonemes)
        # plt.imshow(phoneme_feats[-1].detach().cpu().numpy())
        # plt.savefig('demo2.png')
        # print(phonemes[-1])
        # print(phoneme_feats[-1][:, 0])
        srcmask = get_key_padding_mask(phonemes, pad_idx=self.padding_idx)

        # print(srcmask.shape)
        # plt.imshow(srcmask.detach().cpu().numpy())
        # plt.savefig('demo.png')

        attn_mask = srcmask.unsqueeze(-1).repeat(self.enc_num_head, 1, phoneme_feats.shape[1])
        phoneme_feats, memory = self.encoder(phoneme_feats, src_mask=None, src_key_padding_mask=srcmask)
        # print(srcmask.shape, phoneme_feats.shape,  phoneme_feats)
        # plt.imshow(phoneme_feats[-1].detach().cpu().numpy())
        # plt.savefig('demo.png')
        # exit()
        preddurations = self.durPred(phoneme_feats)
        # print(durations[0])
        spec_feats = upsample(phoneme_feats, durations if durations is not None else preddurations)
        # spec_feats = torch.rand((16, 400, 256)).to(device)

        srcmask = get_key_padding_mask(spec_feats, pad_idx=self.padding_idx)
        # plt.imshow(srcmask.detach().cpu().numpy())
        # plt.savefig('demo3.png')
        #
        # exit()
        attn_mask = srcmask.unsqueeze(-1).repeat(self.dec_num_head, 1, spec_feats.shape[1])
        # print(spec_feats.shape, phoneme_feats.shape, srcmask.shape, attn_mask.shape)
        output_mel_feats, memory, *_ = self.decoder(spec_feats, src_mask=None, src_key_padding_mask=srcmask)
        mel_post = self.linear(output_mel_feats)
        return mel_post, preddurations

def upsample(feats, durs, freq=1):
    return torch.nn.utils.rnn.pad_sequence([torch.repeat_interleave(feats[i], durs[i].long(), dim=0)
                                            for i in range(len(durs))],
                                            batch_first=True, padding_value=0.0)
