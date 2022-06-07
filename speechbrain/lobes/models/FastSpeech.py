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


    def forward(self, tokens, durations=None):
        token_feats = self.encPreNet(tokens)
        srcmask = get_key_padding_mask(tokens, pad_idx=self.padding_idx)
        attn_mask = srcmask.unsqueeze(-1).repeat(self.enc_num_head, 1, token_feats.shape[1])
        token_feats, memory = self.encoder(token_feats, src_mask=None, src_key_padding_mask=srcmask)
        predict_durations = self.durPred(token_feats).squeeze()
        if predict_durations.dim() == 1: predict_durations = predict_durations.unsqueeze(0)
        if durations is None: dur_pred_reverse_log = torch.clamp(torch.exp(predict_durations) - 1, 0)
        spec_feats = upsample(token_feats, durations if durations is not None else dur_pred_reverse_log)
        srcmask = get_key_padding_mask(spec_feats, pad_idx=self.padding_idx)
        attn_mask = srcmask.unsqueeze(-1).repeat(self.dec_num_head, 1, spec_feats.shape[1])
        output_mel_feats, memory, *_ = self.decoder(spec_feats, src_mask=None, src_key_padding_mask=srcmask)
        mel_post = self.linear(output_mel_feats)
        return mel_post, predict_durations

def upsample(feats, durs, freq=1, padding_value=0.0):
    return torch.nn.utils.rnn.pad_sequence([torch.repeat_interleave(feats[i], durs[i].long(), dim=0)
                                            for i in range(len(durs))],
                                            batch_first=True, padding_value=padding_value)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    Arguments
    ---------
    n_frames_per_step: int
        the number of frames per step
    Returns
    -------
    result: tuple
        a tuple of tensors to be used as inputs/targets
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x
        )
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step
        # pdb.set_trace()

    # TODO: Make this more intuitive, use the pipeline
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """
        # TODO: Remove for loops and this dirty hack
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]['mel_text_pair']

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        dur_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        dur_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):

            text = batch[ids_sorted_decreasing[i]][0]

            dur = batch[ids_sorted_decreasing[i]][1]
            # print(text, dur)
            dur_padded[i, : dur.size(0)] = dur
            text_padded[i, : text.size(0)] = text
            # print(dur_padded, text_padded)
        # exit()
        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][2]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]['label'])
            wavs.append(raw_batch[idx]['wav'])
        # count number of items - characters in text
        len_x = [x[3] for x in batch]
        len_x = torch.Tensor(len_x)
        mel_padded = mel_padded.permute(0, 2, 1)
        return (
            text_padded,
            dur_padded,
            input_lengths,
            mel_padded,
            output_lengths,
            len_x,
            labels,
            wavs
        )
