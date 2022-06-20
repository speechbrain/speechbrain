"""
Neural network modules for the FastPitch end-to-end neural
Text-to-Speech (TTS) model
Authors
* Sathvik Udupa 2022
"""

import torch
from torch import nn
import sys
sys.path.append('../../../')
import torch.nn as nn
from torch.nn import functional as F
from speechbrain.nnet.embedding import Embedding
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder, get_key_padding_mask

class EncoderPreNet(nn.Module):
    """Embedding layer for tokens
    Arguments
    ---------
    n_vocab: int
        size of the dictionary of embeddings
    blank_id: int
        padding index
    out_channels: int
        the size of each embedding vector
    device: str
        specify device for embedding
    Example
    -------
    >>> from speechbrain.nnet.embedding import Embedding
    >>> encoder_prenet_layer = EncoderPreNet(n_vocab=40, blank_id=0, out_channels=384)
    >>> x = torch.randn(3, 5)
    >>> y = encoder_prenet_layer(x)
    >>> y.shape
    torch.Size([3, 5, 384])
    """
    def __init__(self, n_vocab, blank_id, out_channels=512,  device='cuda:0'):
        super().__init__()
        self.token_embedding = Embedding(num_embeddings=n_vocab, embedding_dim=out_channels, blank_id=blank_id).to(device)

    def forward(self, x):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            a (batch, tokens) input tensor
        Returns
        -------
        output: torch.Tensor
            the embedding layer output
        """
        x = self.token_embedding(x)
        return x

class DurationPredictor(nn.Module):
    """Duration predictor layer
    Arguments
    ---------
    in_channels: int
       input feature dimension for convolution layers
    out_channels: int
       output feature dimension for convolution layers
    kernel_size: int
       duration predictor convolution kernal size
    Example
    -------
    >>> from speechbrain.lobes.models.FastPitch import FastPitch
    >>> duration_predictor_layer = DurationPredictor(in_channels=384, out_channels=384, kernel_size=3)
    >>> x = torch.randn(3, 400, 384)
    >>> y = duration_predictor_layer(x)
    >>> y.shape
    torch.Size([3, 400, 1])
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.linear = nn.Linear(out_channels, 1)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            a (batch, time_steps, features) input tensor
        Returns
        -------
        output: torch.Tensor
            the duration predictor outputs
        """
        x = self.conv1(x.transpose(1, 2))
        x = self.relu(x)
        x = self.ln1(x.transpose(1, 2)).to(x.dtype)

        x = self.conv2(x.transpose(1, 2))
        x = self.relu(x)
        x = self.ln2(x.transpose(1, 2)).to(x.dtype)

        return self.linear(x)

class FastPitch(nn.Module):
    """The FastPitch text-to-speech model.

    This class is the main entry point for the model, which is responsible
    for instantiating all submodules, which, in turn, manage the individual
    neural network layers

    Simplified STRUCTURE: input->token embedding ->encoder ->duration predictor ->duration
    upsampler -> decoder -> output

    During training, teacher forcing is used (ground truth durations are used for upsampling)

    Arguments
    ---------
    #encoder parameters
    enc_num_layers: int
        number of transformer layers (TransformerEncoderLayer) in encoder
    enc_num_head: int
        number of multi-head-attention (MHA) heads in encoder transformer layers
    enc_d_model: int
        the number of expected features in the encoder
    enc_ffn_dim: int
        the dimension of the feedforward network model
    enc_k_dim: int
        the dimension of the key
    enc_v_dim: int
        the dimension of the value
    enc_dropout: float
        Dropout for the encoder
    normalize_before: bool
        whether normalization should be applied before or after MHA or FFN in Transformer layers.
    ffn_type: str
        whether to use convolutional layers instead of feed forward network inside tranformer layer

    #decoder parameters
    dec_num_layers: int
        number of transformer layers (TransformerEncoderLayer) in decoder
    dec_num_head: int
        number of multi-head-attention (MHA) heads in decoder transformer layers
    dec_d_model: int
        the number of expected features in the decoder
    dec_ffn_dim: int
        the dimension of the feedforward network model
    dec_k_dim: int
        the dimension of the key
    dec_v_dim: int
        the dimension of the value
    dec_dropout: float
        dropout for the decoder
    normalize_before: bool
        whether normalization should be applied before or after MHA or FFN in Transformer layers.
    ffn_type: str
        whether to use convolutional layers instead of feed forward network inside tranformer layer

    #data io
    n_char: int
        the number of symbols for the token embedding
    n_mels: int
        number of bins in mel spectrogram
    padding_idx: int
        the index for padding
    dur_pred_kernel_size: int
        the convolution kernel size in duration predictor

    Example
    -------
    >>> import torch
    >>> _ = torch.manual_seed(213312)
    >>> from speechbrain.lobes.models.FastPitch import FastPitch
    >>> model = FastPitch(
    ...    enc_num_layers=6,
    ...    enc_num_head=2,
    ...    enc_d_model=384,
    ...    enc_ffn_dim=1536,
    ...    enc_k_dim=384,
    ...    enc_v_dim=384,
    ...    enc_dropout=0.1,
    ...    dec_num_layers=6,
    ...    dec_num_head=2,
    ...    dec_d_model=384,
    ...    dec_ffn_dim=1536,
    ...    dec_k_dim=384,
    ...    dec_v_dim=384,
    ...    dec_dropout=0.1,
    ...    normalize_before=False,
    ...    ffn_type=1dcnn,
    ...    n_char=40,
    ...    n_mels=80,
    ...    padding_idx=0,
    ...    dur_pred_kernel_size=3)
    >>> inputs = torch.tensor([
    ...     [13, 12, 31, 14, 19],
    ...     [31, 16, 30, 31, 0],
    ... ])
    >>> input_lengths = torch.tensor([5, 4])
    >>> mel_post, predict_durations = model(inputs, durations=None)
    >>> mel_post.shape, predict_durations.shape
    (torch.Size([2, 96, 80]), torch.Size([2, 5]))
    """
    def __init__(self, enc_num_layers,
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
                    dur_pred_kernel_size,
                    pitch_pred_kernel_size):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx

        self.encPreNet = EncoderPreNet(n_char, padding_idx, out_channels=enc_d_model)
        self.durPred = DurationPredictor(in_channels=enc_d_model, out_channels=enc_d_model, kernel_size=dur_pred_kernel_size)
        self.pitchPred = DurationPredictor(in_channels=enc_d_model, out_channels=enc_d_model, kernel_size=dur_pred_kernel_size)
        self.pitchEmbed = nn.Conv1d(1, enc_d_model, pitch_pred_kernel_size, padding=pitch_pred_kernel_size//2)
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


    def forward(self, tokens, durations=None, pitch=None, pace=1.0):
        """forward pass for training and inference
        Arguments
        ---------
        tokens: torch.tensor
            batch of input tokens
        durations: torch.tensor
            batch of durations for each token. If it is None, the model will infer on predicted durations
        Returns
        ---------
        mel_post: torch.Tensor
            mel outputs from the decoder
        predict_durations: torch.Tensor
            predicted durations for each token
        """
        token_feats = self.encPreNet(tokens)
        srcmask = get_key_padding_mask(tokens, pad_idx=self.padding_idx)
        # print(srcmask)
        # exit()
        attn_mask = srcmask.unsqueeze(-1).repeat(self.enc_num_head, 1, token_feats.shape[1])
        token_feats, memory = self.encoder(token_feats, src_mask=None, src_key_padding_mask=srcmask)
        predict_durations = self.durPred(token_feats).squeeze()
        predict_pitch = self.pitchPred(token_feats).transpose(2, 1)
        if pitch is not None:
            cum_sum_durations_end = torch.cumsum(durations, dim=1)
            cum_sum_durations_begin = torch.sub(cum_sum_durations_end, durations)
            cum_sum_durations_end = torch.sub(cum_sum_durations_end ,1)*~srcmask
            cum_sum_durations_begin = torch.sub(cum_sum_durations_begin ,1)*~srcmask
            cum_sum_durations_begin[:, 0] = 0

            pitch_per_token = torch.sub(torch.gather(pitch, -1, cum_sum_durations_end),
                                torch.gather(pitch, -1, cum_sum_durations_begin))

            pitch_per_token = torch.where(durations==0.0, pitch_per_token, pitch_per_token/durations)
            pitch_per_token = pitch_per_token*~srcmask
            pitch_per_token = pitch_per_token.unsqueeze(1)
            pitch = self.pitchEmbed(pitch_per_token).transpose(2, 1)
        else:
            pitch = self.pitchEmbed(predict_pitch).transpose(2, 1)
            pitch_per_token = None
        token_feats = token_feats.add(pitch)
        
        if predict_durations.dim() == 1: predict_durations = predict_durations.unsqueeze(0)
        if durations is None: dur_pred_reverse_log = torch.clamp(torch.exp(predict_durations) - 1, 0)
        spec_feats = upsample(token_feats, durations if durations is not None else dur_pred_reverse_log, pace=pace)
        
        
        srcmask = get_key_padding_mask(spec_feats, pad_idx=self.padding_idx)
        attn_mask = srcmask.unsqueeze(-1).repeat(self.dec_num_head, 1, spec_feats.shape[1])
        output_mel_feats, memory, *_ = self.decoder(spec_feats, src_mask=None, src_key_padding_mask=srcmask)

        mel_post = self.linear(output_mel_feats)
        return mel_post, predict_durations, predict_pitch, pitch_per_token

def upsample(feats, durs, pace=1.0, padding_value=0.0):
    """upsample encoder ouput according to durations
    Arguments
    ---------
    feats: torch.tensor
        batch of input tokens
    durs: torch.tensor
        durations to be used to upsample
    pace: int
        scaling factor for durations
    padding_value: int
        padding index
    Returns
    ---------
    mel_post: torch.Tensor
        mel outputs from the decoder
    predict_durations: torch.Tensor
        predicted durations for each token
    """
    return torch.nn.utils.rnn.pad_sequence([torch.repeat_interleave(feats[i], (pace*durs[i]).long(), dim=0)
                                            for i in range(len(durs))],
                                            batch_first=True, padding_value=padding_value)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step

    result: tuple
        a tuple of tensors to be used as inputs/targets
        (
            text_padded,
            dur_padded,
            input_lengths,
            mel_padded,
            output_lengths,
            len_x,
            labels,
            wavs
        )
    """


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
        pitch_padded = torch.FloatTensor(len(batch), max_target_len)
        pitch_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][2]
            pitch = batch[idx][3]
            mel_padded[i, :, : mel.size(1)] = mel
            pitch_padded[i, :pitch.size(0)] = pitch
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]['label'])
            wavs.append(raw_batch[idx]['wav'])
        # count number of items - characters in text
        len_x = [x[4] for x in batch]
        len_x = torch.Tensor(len_x)
        mel_padded = mel_padded.permute(0, 2, 1)
        return (
            text_padded,
            dur_padded,
            input_lengths,
            mel_padded,
            pitch_padded,
            output_lengths,
            len_x,
            labels,
            wavs
        )


class Loss(nn.Module):
   
    def __init__(
        self,
        log_scale_durations,
        duration_loss_weight,
        pitch_loss_weight,
        mel_loss_weight
    ):
        super().__init__()

        self.mel_loss = nn.L1Loss()
        self.dur_loss = nn.L1Loss()
        self.pitch_loss = nn.L1Loss()
        self.log_scale_durations = log_scale_durations
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight

    def forward(
        self, predictions, targets):
        """Computes the value of the loss function and updates stats
        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth data
        Returns
        -------
        loss: torch.Tensor
            the loss value
        """
        mel_target, target_durations, target_pitch, mel_length, phon_len  = targets
        assert len(mel_target.shape) == 3
        mel_out, log_durations, predicted_pitch, token_target_pitch = predictions
        predicted_pitch = predicted_pitch.squeeze()
        token_target_pitch = token_target_pitch.squeeze()
        log_durations = log_durations.squeeze()
        if self.log_scale_durations:
            log_target_durations = torch.log(target_durations.float() + 1)
            durations = torch.clamp(torch.exp(log_durations) - 1, 0, 20)
        mel_loss, dur_loss = 0, 0
        #change this to perform batch level using padding mask
        for i in range(mel_target.shape[0]):
            if i == 0:
                mel_loss = self.mel_loss(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
                dur_loss = self.dur_loss(log_durations[i, :phon_len[i]], log_target_durations[i, :phon_len[i]].to(torch.float32))
                pitch_loss = self.pitch_loss(predicted_pitch[i, :phon_len[i]], token_target_pitch[i, :phon_len[i]].to(torch.float32))
            else:
                mel_loss = mel_loss + self.mel_loss(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
                dur_loss = dur_loss + self.dur_loss(log_durations[i, :phon_len[i]], log_target_durations[i, :phon_len[i]].to(torch.float32))
                pitch_loss = pitch_loss + self.pitch_loss(predicted_pitch[i, :phon_len[i]], token_target_pitch[i, :phon_len[i]].to(torch.float32))
        mel_loss = torch.div(mel_loss, len(mel_target))
        dur_loss = torch.div(dur_loss, len(mel_target))
        pitch_loss = torch.div(pitch_loss, len(mel_target))
        return mel_loss*self.mel_loss_weight + dur_loss*self.duration_loss_weight + pitch_loss*self.pitch_loss_weight