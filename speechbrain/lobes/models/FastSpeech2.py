"""
Neural network modules for the FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
synthesis model
Authors
* Sathvik Udupa 2022
* Pradnya Kandarkar 2023
* Yingzhi Wang 2023
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from speechbrain.nnet import CNN, linear
from speechbrain.nnet.embedding import Embedding
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    get_key_padding_mask,
)
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.losses import bce_loss


class PositionalEmbedding(nn.Module):
    """Computation of the positional embeddings.
    Arguments
    ---------
    embed_dim: int
        dimensionality of the embeddings.
    """

    def __init__(self, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.demb = embed_dim
        inv_freq = 1 / (
            10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, mask, dtype):
        """Computes the forward pass
        Arguments
        ---------
        seq_len: int
            length of the sequence
        mask: torch.tensor
            mask applied to the positional embeddings
        dtype: str
            dtype of the embeddings
        Returns
        -------
        pos_emb: torch.Tensor
            the tensor with positional embeddings
        """
        pos_seq = torch.arange(seq_len, device=mask.device).to(dtype)

        sinusoid_inp = torch.matmul(
            torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0)
        )
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        return pos_emb[None, :, :] * mask[:, :, None]


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
    Example
    -------
    >>> from speechbrain.nnet.embedding import Embedding
    >>> from speechbrain.lobes.models.FastSpeech2 import EncoderPreNet
    >>> encoder_prenet_layer = EncoderPreNet(n_vocab=40, blank_id=0, out_channels=384)
    >>> x = torch.rand(3, 5)
    >>> y = encoder_prenet_layer(x)
    >>> y.shape
    torch.Size([3, 5, 384])
    """

    def __init__(self, n_vocab, blank_id, out_channels=512):
        super().__init__()
        self.token_embedding = Embedding(
            num_embeddings=n_vocab,
            embedding_dim=out_channels,
            blank_id=blank_id,
        )

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
        self.token_embedding = self.token_embedding.to(x.device)
        x = self.token_embedding(x)
        return x


class PostNet(nn.Module):
    """
    FastSpeech2 Conv Postnet
    Arguments
    ---------
    n_mel_channels: int
       input feature dimension for convolution layers
    postnet_embedding_dim: int
       output feature dimension for convolution layers
    postnet_kernel_size: int
       postnet convolution kernal size
    postnet_n_convolutions: int
       number of convolution layers
    postnet_dropout: float
        dropout probability fot postnet
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        postnet_dropout=0.5,
    ):
        super(PostNet, self).__init__()
        self.conv_pre = CNN.Conv1d(
            in_channels=n_mel_channels,
            out_channels=postnet_embedding_dim,
            kernel_size=postnet_kernel_size,
            padding="same",
        )

        self.convs_intermedite = nn.ModuleList()
        for i in range(1, postnet_n_convolutions - 1):
            self.convs_intermedite.append(
                CNN.Conv1d(
                    in_channels=postnet_embedding_dim,
                    out_channels=postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding="same",
                ),
            )

        self.conv_post = CNN.Conv1d(
            in_channels=postnet_embedding_dim,
            out_channels=n_mel_channels,
            kernel_size=postnet_kernel_size,
            padding="same",
        )

        self.tanh = nn.Tanh()
        self.ln1 = nn.LayerNorm(postnet_embedding_dim)
        self.ln2 = nn.LayerNorm(postnet_embedding_dim)
        self.ln3 = nn.LayerNorm(n_mel_channels)
        self.dropout1 = nn.Dropout(postnet_dropout)
        self.dropout2 = nn.Dropout(postnet_dropout)
        self.dropout3 = nn.Dropout(postnet_dropout)

    def forward(self, x):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            a (batch, time_steps, features) input tensor
        Returns
        -------
        output: torch.Tensor
            the spectrogram predicted
        """
        x = self.conv_pre(x)
        x = self.ln1(x).to(x.dtype)
        x = self.tanh(x)
        x = self.dropout1(x)

        for i in range(len(self.convs_intermedite)):
            x = self.convs_intermedite[i](x)
        x = self.ln2(x).to(x.dtype)
        x = self.tanh(x)
        x = self.dropout2(x)

        x = self.conv_post(x)
        x = self.ln3(x).to(x.dtype)
        x = self.dropout3(x)

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
    dropout: float
       dropout probability, 0 by default
    Example
    -------
    >>> from speechbrain.lobes.models.FastSpeech2 import FastSpeech2
    >>> duration_predictor_layer = DurationPredictor(in_channels=384, out_channels=384, kernel_size=3)
    >>> x = torch.randn(3, 400, 384)
    >>> mask = torch.ones(3, 400, 384)
    >>> y = duration_predictor_layer(x, mask)
    >>> y.shape
    torch.Size([3, 400, 1])
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, dropout=0.0, n_units=1
    ):
        super().__init__()
        self.conv1 = CNN.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv2 = CNN.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.linear = linear.Linear(n_neurons=n_units, input_size=out_channels)
        self.ln1 = LayerNorm(out_channels)
        self.ln2 = LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            a (batch, time_steps, features) input tensor
        x_mask: torch.Tensor
            mask of input tensor
        Returns
        -------
        output: torch.Tensor
            the duration predictor outputs
        """
        x = self.relu(self.conv1(x * x_mask))
        x = self.ln1(x).to(x.dtype)
        x = self.dropout1(x)

        x = self.relu(self.conv2(x * x_mask))
        x = self.ln2(x).to(x.dtype)
        x = self.dropout2(x)

        return self.linear(x * x_mask)


class SPNPredictor(nn.Module):

    """
    This module for the silent phoneme predictor. It receives phoneme sequences without any silent phoneme token as
    input and predicts whether a silent phoneme should be inserted after a position. This is to avoid the issue of fast
    pace at inference time due to having no silent phoneme tokens in the input sequence.

    Arguments
    ---------
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
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn
    n_char: int
        the number of symbols for the token embedding
    padding_idx: int
        the index for padding
    """

    def __init__(
        self,
        enc_num_layers,
        enc_num_head,
        enc_d_model,
        enc_ffn_dim,
        enc_k_dim,
        enc_v_dim,
        enc_dropout,
        normalize_before,
        ffn_type,
        ffn_cnn_kernel_size_list,
        n_char,
        padding_idx,
    ):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.padding_idx = padding_idx

        self.encPreNet = EncoderPreNet(
            n_char, padding_idx, out_channels=enc_d_model
        )

        self.sinusoidal_positional_embed_encoder = PositionalEmbedding(
            enc_d_model
        )

        self.spn_encoder = TransformerEncoder(
            num_layers=enc_num_layers,
            nhead=enc_num_head,
            d_ffn=enc_ffn_dim,
            d_model=enc_d_model,
            kdim=enc_k_dim,
            vdim=enc_v_dim,
            dropout=enc_dropout,
            activation=nn.ReLU,
            normalize_before=normalize_before,
            ffn_type=ffn_type,
            ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
        )

        self.spn_linear = linear.Linear(n_neurons=1, input_size=enc_d_model)

    def forward(self, tokens, last_phonemes):
        """forward pass for the module
        Arguments
        ---------
        tokens: torch.Tensor
            input tokens without silent phonemes
        last_phonemes: torch.Tensor
            indicates if a phoneme at an index is the last phoneme of a word or not

        Returns
        ---------
        spn_decision: torch.Tensor
            indicates if a silent phoneme should be inserted after a phoneme
        """
        token_feats = self.encPreNet(tokens)
        last_phonemes = torch.unsqueeze(last_phonemes, 2).repeat(
            1, 1, token_feats.shape[2]
        )

        token_feats = token_feats + last_phonemes

        srcmask = get_key_padding_mask(tokens, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1)
        pos = self.sinusoidal_positional_embed_encoder(
            token_feats.shape[1], srcmask, token_feats.dtype
        )
        token_feats = torch.add(token_feats, pos) * srcmask_inverted

        spn_mask = (
            torch.triu(
                torch.ones(
                    token_feats.shape[1],
                    token_feats.shape[1],
                    device=token_feats.device,
                ),
                diagonal=1,
            )
            .bool()
            .repeat(self.enc_num_head * token_feats.shape[0], 1, 1)
        )

        spn_token_feats, _ = self.spn_encoder(
            token_feats, src_mask=spn_mask, src_key_padding_mask=srcmask
        )
        spn_decision = self.spn_linear(spn_token_feats).squeeze()

        return spn_decision

    def infer(self, tokens, last_phonemes):
        """inference function
        Arguments
        ---------
        tokens: torch.Tensor
            input tokens without silent phonemes
        last_phonemes: torch.Tensor
            indicates if a phoneme at an index is the last phoneme of a word or not

        Returns
        ---------
        spn_decision: torch.Tensor
            indicates if a silent phoneme should be inserted after a phoneme
        """
        spn_decision = self.forward(tokens, last_phonemes)
        spn_decision = torch.sigmoid(spn_decision) > 0.8
        return spn_decision


class FastSpeech2(nn.Module):
    """The FastSpeech2 text-to-speech model.
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
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn
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
        whether to use convolutional layers instead of feed forward network inside tranformer layer.
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn
    n_char: int
        the number of symbols for the token embedding
    n_mels: int
        number of bins in mel spectrogram
    postnet_embedding_dim: int
       output feature dimension for convolution layers
    postnet_kernel_size: int
       postnet convolution kernal size
    postnet_n_convolutions: int
       number of convolution layers
    postnet_dropout: float
        dropout probability fot postnet
    padding_idx: int
        the index for padding
    dur_pred_kernel_size: int
        the convolution kernel size in duration predictor
    pitch_pred_kernel_size: int
        kernel size for pitch prediction.
    energy_pred_kernel_size: int
        kernel size for energy prediction.
    variance_predictor_dropout: float
        dropout probability for variance predictor (duration/pitch/energy)
    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.FastSpeech2 import FastSpeech2
    >>> model = FastSpeech2(
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
    ...    ffn_type='1dcnn',
    ...    ffn_cnn_kernel_size_list=[9, 1],
    ...    n_char=40,
    ...    n_mels=80,
    ...    postnet_embedding_dim=512,
    ...    postnet_kernel_size=5,
    ...    postnet_n_convolutions=5,
    ...    postnet_dropout=0.5,
    ...    padding_idx=0,
    ...    dur_pred_kernel_size=3,
    ...    pitch_pred_kernel_size=3,
    ...    energy_pred_kernel_size=3,
    ...    variance_predictor_dropout=0.5)
    >>> inputs = torch.tensor([
    ...     [13, 12, 31, 14, 19],
    ...     [31, 16, 30, 31, 0],
    ... ])
    >>> input_lengths = torch.tensor([5, 4])
    >>> durations = torch.tensor([
    ...     [2, 4, 1, 5, 3],
    ...     [1, 2, 4, 3, 0],
    ... ])
    >>> mel_post, postnet_output, predict_durations, predict_pitch, avg_pitch, predict_energy, avg_energy, mel_lens = model(inputs, durations=durations)
    >>> mel_post.shape, predict_durations.shape
    (torch.Size([2, 15, 80]), torch.Size([2, 5]))
    >>> predict_pitch.shape, predict_energy.shape
    (torch.Size([2, 5, 1]), torch.Size([2, 5, 1]))
    """

    def __init__(
        self,
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
        ffn_cnn_kernel_size_list,
        n_char,
        n_mels,
        postnet_embedding_dim,
        postnet_kernel_size,
        postnet_n_convolutions,
        postnet_dropout,
        padding_idx,
        dur_pred_kernel_size,
        pitch_pred_kernel_size,
        energy_pred_kernel_size,
        variance_predictor_dropout,
    ):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx
        self.sinusoidal_positional_embed_encoder = PositionalEmbedding(
            enc_d_model
        )
        self.sinusoidal_positional_embed_decoder = PositionalEmbedding(
            dec_d_model
        )

        self.encPreNet = EncoderPreNet(
            n_char, padding_idx, out_channels=enc_d_model
        )
        self.durPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.pitchPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.energyPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.pitchEmbed = CNN.Conv1d(
            in_channels=1,
            out_channels=enc_d_model,
            kernel_size=pitch_pred_kernel_size,
            padding="same",
            skip_transpose=True,
        )

        self.energyEmbed = CNN.Conv1d(
            in_channels=1,
            out_channels=enc_d_model,
            kernel_size=energy_pred_kernel_size,
            padding="same",
            skip_transpose=True,
        )
        self.encoder = TransformerEncoder(
            num_layers=enc_num_layers,
            nhead=enc_num_head,
            d_ffn=enc_ffn_dim,
            d_model=enc_d_model,
            kdim=enc_k_dim,
            vdim=enc_v_dim,
            dropout=enc_dropout,
            activation=nn.ReLU,
            normalize_before=normalize_before,
            ffn_type=ffn_type,
            ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
        )

        self.decoder = TransformerEncoder(
            num_layers=dec_num_layers,
            nhead=dec_num_head,
            d_ffn=dec_ffn_dim,
            d_model=dec_d_model,
            kdim=dec_k_dim,
            vdim=dec_v_dim,
            dropout=dec_dropout,
            activation=nn.ReLU,
            normalize_before=normalize_before,
            ffn_type=ffn_type,
            ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
        )

        self.linear = linear.Linear(n_neurons=n_mels, input_size=dec_d_model)
        self.postnet = PostNet(
            n_mel_channels=n_mels,
            postnet_embedding_dim=postnet_embedding_dim,
            postnet_kernel_size=postnet_kernel_size,
            postnet_n_convolutions=postnet_n_convolutions,
            postnet_dropout=postnet_dropout,
        )

    def forward(
        self,
        tokens,
        durations=None,
        pitch=None,
        energy=None,
        pace=1.0,
        pitch_rate=1.0,
        energy_rate=1.0,
    ):
        """forward pass for training and inference
        Arguments
        ---------
        tokens: torch.Tensor
            batch of input tokens
        durations: torch.Tensor
            batch of durations for each token. If it is None, the model will infer on predicted durations
        pitch: torch.Tensor
            batch of pitch for each frame. If it is None, the model will infer on predicted pitches
        energy: torch.Tensor
            batch of energy for each frame. If it is None, the model will infer on predicted energies
        pace: float
            scaling factor for durations
        pitch_rate: float
            scaling factor for pitches
        energy_rate: float
            scaling factor for energies
        Returns
        ---------
        mel_post: torch.Tensor
            mel outputs from the decoder
        postnet_output: torch.Tensor
            mel outputs from the postnet
        predict_durations: torch.Tensor
            predicted durations of each token
        predict_pitch: torch.Tensor
            predicted pitches of each token
        avg_pitch: torch.Tensor
            target pitches for each token if input pitch is not None
            None if input pitch is None
        predict_energy: torch.Tensor
            predicted energies of each token
        avg_energy: torch.Tensor
            target energies for each token if input energy is not None
            None if input energy is None
        mel_length:
            predicted lengths of mel spectrograms
        """
        srcmask = get_key_padding_mask(tokens, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1)

        # prenet & encoder
        token_feats = self.encPreNet(tokens)
        pos = self.sinusoidal_positional_embed_encoder(
            token_feats.shape[1], srcmask, token_feats.dtype
        )
        token_feats = torch.add(token_feats, pos) * srcmask_inverted
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.enc_num_head, 1, token_feats.shape[1])
            .permute(0, 2, 1)
            .bool()
        )
        token_feats, _ = self.encoder(
            token_feats, src_mask=attn_mask, src_key_padding_mask=srcmask
        )
        token_feats = token_feats * srcmask_inverted

        # duration predictor
        predict_durations = self.durPred(
            token_feats, srcmask_inverted
        ).squeeze()

        if predict_durations.dim() == 1:
            predict_durations = predict_durations.unsqueeze(0)
        if durations is None:
            dur_pred_reverse_log = torch.clamp(
                torch.exp(predict_durations) - 1, 0
            )

        # pitch predictor
        avg_pitch = None
        predict_pitch = self.pitchPred(token_feats, srcmask_inverted)
        # use a pitch rate to adjust the pitch
        predict_pitch = predict_pitch * pitch_rate
        if pitch is not None:
            avg_pitch = average_over_durations(pitch.unsqueeze(1), durations)
            pitch = self.pitchEmbed(avg_pitch)
            avg_pitch = avg_pitch.permute(0, 2, 1)
        else:
            pitch = self.pitchEmbed(predict_pitch.permute(0, 2, 1))
        pitch = pitch.permute(0, 2, 1)
        token_feats = token_feats.add(pitch)

        # energy predictor
        avg_energy = None
        predict_energy = self.energyPred(token_feats, srcmask_inverted)
        # use an energy rate to adjust the energy
        predict_energy = predict_energy * energy_rate
        if energy is not None:
            avg_energy = average_over_durations(energy.unsqueeze(1), durations)
            energy = self.energyEmbed(avg_energy)
            avg_energy = avg_energy.permute(0, 2, 1)
        else:
            energy = self.energyEmbed(predict_energy.permute(0, 2, 1))
        energy = energy.permute(0, 2, 1)
        token_feats = token_feats.add(energy)

        # upsamples the durations
        spec_feats, mel_lens = upsample(
            token_feats,
            durations if durations is not None else dur_pred_reverse_log,
            pace=pace,
        )
        srcmask = get_key_padding_mask(spec_feats, pad_idx=self.padding_idx)
        srcmask_inverted = (~srcmask).unsqueeze(-1)
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.dec_num_head, 1, spec_feats.shape[1])
            .permute(0, 2, 1)
            .bool()
        )

        # decoder
        pos = self.sinusoidal_positional_embed_decoder(
            spec_feats.shape[1], srcmask, spec_feats.dtype
        )

        spec_feats = torch.add(spec_feats, pos) * srcmask_inverted

        output_mel_feats, memory, *_ = self.decoder(
            spec_feats, src_mask=attn_mask, src_key_padding_mask=srcmask
        )

        # postnet
        mel_post = self.linear(output_mel_feats) * srcmask_inverted
        postnet_output = self.postnet(mel_post) + mel_post
        return (
            mel_post,
            postnet_output,
            predict_durations,
            predict_pitch,
            avg_pitch,
            predict_energy,
            avg_energy,
            torch.tensor(mel_lens),
        )


def average_over_durations(values, durs):
    """Average values over durations.
    Arguments
    ---------
    values: torch.Tensor
        shape: [B, 1, T_de]
    durs: torch.Tensor
        shape: [B, T_en]
    Returns
    ---------
    avg: torch.Tensor
        shape: [B, 1, T_en]
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(
        torch.cumsum(values != 0.0, dim=2), (1, 0)
    )
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, length = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, length)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, length)

    values_sums = (
        torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
    ).float()
    values_nelems = (
        torch.gather(values_nonzero_cums, 2, dce)
        - torch.gather(values_nonzero_cums, 2, dcs)
    ).float()

    avg = torch.where(
        values_nelems == 0.0, values_nelems, values_sums / values_nelems
    )
    return avg


def upsample(feats, durs, pace=1.0, padding_value=0.0):
    """upsample encoder ouput according to durations
    Arguments
    ---------
    feats: torch.tensor
        batch of input tokens
    durs: torch.tensor
        durations to be used to upsample
    pace: float
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
    upsampled_mels = [
        torch.repeat_interleave(feats[i], (pace * durs[i]).long(), dim=0)
        for i in range(len(durs))
    ]

    mel_lens = [mel.shape[0] for mel in upsampled_mels]

    padded_upsampled_mels = torch.nn.utils.rnn.pad_sequence(
        upsampled_mels, batch_first=True, padding_value=padding_value,
    )
    return padded_upsampled_mels, mel_lens


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
        # TODO: Remove for loops
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]["mel_text_pair"]

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        # Get max_no_spn_seq_len
        no_spn_seq_lengths, no_spn_ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[-2]) for x in batch]),
            dim=0,
            descending=True,
        )
        max_no_spn_seq_len = no_spn_seq_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        no_spn_seq_padded = torch.LongTensor(len(batch), max_no_spn_seq_len)
        last_phonemes_padded = torch.LongTensor(len(batch), max_no_spn_seq_len)
        dur_padded = torch.LongTensor(len(batch), max_input_len)
        spn_labels_padded = torch.FloatTensor(len(batch), max_no_spn_seq_len)
        text_padded.zero_()
        no_spn_seq_padded.zero_()
        last_phonemes_padded.zero_()
        dur_padded.zero_()
        spn_labels_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            no_spn_seq = batch[ids_sorted_decreasing[i]][-2]
            last_phonemes = torch.LongTensor(
                batch[ids_sorted_decreasing[i]][-3]
            )
            dur = batch[ids_sorted_decreasing[i]][1]
            spn_labels = torch.LongTensor(batch[ids_sorted_decreasing[i]][-1])

            text_padded[i, : text.size(0)] = text
            no_spn_seq_padded[i, : no_spn_seq.size(0)] = no_spn_seq
            last_phonemes_padded[i, : last_phonemes.size(0)] = last_phonemes
            dur_padded[i, : dur.size(0)] = dur
            spn_labels_padded[i, : spn_labels.size(0)] = spn_labels

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        pitch_padded = torch.FloatTensor(len(batch), max_target_len)
        pitch_padded.zero_()
        energy_padded = torch.FloatTensor(len(batch), max_target_len)
        energy_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][2]
            pitch = batch[idx][3]
            energy = batch[idx][4]
            mel_padded[i, :, : mel.size(1)] = mel
            pitch_padded[i, : pitch.size(0)] = pitch
            energy_padded[i, : energy.size(0)] = energy
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]["label"])
            wavs.append(raw_batch[idx]["wav"])
        # count number of items - characters in text
        len_x = [x[5] for x in batch]
        len_x = torch.Tensor(len_x)
        mel_padded = mel_padded.permute(0, 2, 1)

        return (
            text_padded,
            dur_padded,
            input_lengths,
            mel_padded,
            pitch_padded,
            energy_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
            no_spn_seq_padded,
            spn_labels_padded,
            last_phonemes_padded,
        )


class Loss(nn.Module):
    """Loss Computation
    Arguments
    ---------
    log_scale_durations: bool
       applies logarithm to target durations
    duration_loss_weight: int
       weight for the duration loss
    pitch_loss_weight: int
       weight for the pitch loss
    energy_loss_weight: int
       weight for the energy loss
    mel_loss_weight: int
       weight for the mel loss
    postnet_mel_loss_weight: int
       weight for the postnet mel loss
    """

    def __init__(
        self,
        log_scale_durations,
        ssim_loss_weight,
        duration_loss_weight,
        pitch_loss_weight,
        energy_loss_weight,
        mel_loss_weight,
        postnet_mel_loss_weight,
        spn_loss_weight=1.0,
        spn_loss_max_epochs=8,
    ):
        super().__init__()

        self.ssim_loss = SSIMLoss()
        self.mel_loss = nn.MSELoss()
        self.postnet_mel_loss = nn.MSELoss()
        self.dur_loss = nn.MSELoss()
        self.pitch_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
        self.log_scale_durations = log_scale_durations
        self.ssim_loss_weight = ssim_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.postnet_mel_loss_weight = postnet_mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.spn_loss_weight = spn_loss_weight
        self.spn_loss_max_epochs = spn_loss_max_epochs

    def forward(self, predictions, targets, current_epoch):
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
        (
            mel_target,
            target_durations,
            target_pitch,
            target_energy,
            mel_length,
            phon_len,
            spn_labels,
        ) = targets
        assert len(mel_target.shape) == 3
        (
            mel_out,
            postnet_mel_out,
            log_durations,
            predicted_pitch,
            average_pitch,
            predicted_energy,
            average_energy,
            mel_lens,
            spn_preds,
        ) = predictions

        predicted_pitch = predicted_pitch.squeeze()
        predicted_energy = predicted_energy.squeeze()

        target_pitch = average_pitch.squeeze()
        target_energy = average_energy.squeeze()

        log_durations = log_durations.squeeze()
        if self.log_scale_durations:
            log_target_durations = torch.log(target_durations.float() + 1)
        # change this to perform batch level using padding mask

        for i in range(mel_target.shape[0]):
            if i == 0:
                mel_loss = self.mel_loss(
                    mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                postnet_mel_loss = self.postnet_mel_loss(
                    postnet_mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                dur_loss = self.dur_loss(
                    log_durations[i, : phon_len[i]],
                    log_target_durations[i, : phon_len[i]].to(torch.float32),
                )
                pitch_loss = self.pitch_loss(
                    predicted_pitch[i, : mel_length[i]],
                    target_pitch[i, : mel_length[i]].to(torch.float32),
                )
                energy_loss = self.energy_loss(
                    predicted_energy[i, : mel_length[i]],
                    target_energy[i, : mel_length[i]].to(torch.float32),
                )
            else:
                mel_loss = mel_loss + self.mel_loss(
                    mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                postnet_mel_loss = postnet_mel_loss + self.postnet_mel_loss(
                    postnet_mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                dur_loss = dur_loss + self.dur_loss(
                    log_durations[i, : phon_len[i]],
                    log_target_durations[i, : phon_len[i]].to(torch.float32),
                )
                pitch_loss = pitch_loss + self.pitch_loss(
                    predicted_pitch[i, : mel_length[i]],
                    target_pitch[i, : mel_length[i]].to(torch.float32),
                )
                energy_loss = energy_loss + self.energy_loss(
                    predicted_energy[i, : mel_length[i]],
                    target_energy[i, : mel_length[i]].to(torch.float32),
                )
        ssim_loss = self.ssim_loss(mel_out, mel_target, mel_length)
        mel_loss = torch.div(mel_loss, len(mel_target))
        postnet_mel_loss = torch.div(postnet_mel_loss, len(mel_target))
        dur_loss = torch.div(dur_loss, len(mel_target))
        pitch_loss = torch.div(pitch_loss, len(mel_target))
        energy_loss = torch.div(energy_loss, len(mel_target))

        spn_loss = bce_loss(spn_preds, spn_labels)
        if current_epoch > self.spn_loss_max_epochs:
            self.spn_loss_weight = 0

        total_loss = (
            ssim_loss * self.ssim_loss_weight
            + mel_loss * self.mel_loss_weight
            + postnet_mel_loss * self.postnet_mel_loss_weight
            + dur_loss * self.duration_loss_weight
            + pitch_loss * self.pitch_loss_weight
            + energy_loss * self.energy_loss_weight
            + spn_loss * self.spn_loss_weight
        )

        loss = {
            "total_loss": total_loss,
            "ssim_loss": ssim_loss * self.ssim_loss_weight,
            "mel_loss": mel_loss * self.mel_loss_weight,
            "postnet_mel_loss": postnet_mel_loss * self.postnet_mel_loss_weight,
            "dur_loss": dur_loss * self.duration_loss_weight,
            "pitch_loss": pitch_loss * self.pitch_loss_weight,
            "energy_loss": energy_loss * self.energy_loss_weight,
            "spn_loss": spn_loss * self.spn_loss_weight,
        }
        return loss


def mel_spectogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    min_max_energy_norm,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal
    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.tensor
        input audio signal
    """
    from torchaudio import transforms

    audio_to_mel = transforms.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        power=power,
        normalized=normalized,
    ).to(audio.device)

    mel_scale = transforms.MelScale(
        sample_rate=sample_rate,
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)
    spec = audio_to_mel(audio)
    mel = mel_scale(spec)
    assert mel.dim() == 2
    assert mel.shape[0] == n_mels
    rmse = torch.norm(mel, dim=0)

    if min_max_energy_norm:
        rmse = (rmse - torch.min(rmse)) / (torch.max(rmse) - torch.min(rmse))

    if compression:
        mel = dynamic_range_compression(mel)

    return mel, rmse


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


class SSIMLoss(torch.nn.Module):
    """SSIM loss as (1 - SSIM)
    SSIM is explained here https://en.wikipedia.org/wiki/Structural_similarity
    """

    def __init__(self):
        super().__init__()
        self.loss_func = _SSIMLoss()

    # from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    def sequence_mask(self, sequence_length, max_len=None):
        """Create a sequence mask for filtering padding in a sequence tensor.
        Arguments
        ---------
        sequence_length: torch.Tensor
            Sequence lengths.
        max_len: int
            Maximum sequence length. Defaults to None.
        Returns
        ---------
        mask: [B, T_max]
        """
        if max_len is None:
            max_len = sequence_length.data.max()
        seq_range = torch.arange(
            max_len, dtype=sequence_length.dtype, device=sequence_length.device
        )
        # B x T_max
        mask = seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)
        return mask

    def sample_wise_min_max(self, x: torch.Tensor, mask: torch.Tensor):
        """Min-Max normalize tensor through first dimension
        Arguments
        ---------
        x: torch.Tensor
            input tensor [B, D1, D2]
        m: torch.Tensor
            input mask [B, D1, 1]
        """
        maximum = torch.amax(x.masked_fill(~mask, 0), dim=(1, 2), keepdim=True)
        minimum = torch.amin(
            x.masked_fill(~mask, 1e30), dim=(1, 2), keepdim=True
        )
        return (x - minimum) / (maximum - minimum + 1e-8)

    def forward(self, y_hat, y, length):
        """
        Arguments
        ---------
        y_hat: torch.Tensor
            model prediction values [B, T, D].
        y: torch.Tensor
            target values [B, T, D].
        length: torch.Tensor
            length of each sample in a batch for masking.
        Returns
        ---------
        loss: Average loss value in range [0, 1] masked by the length.
        """
        mask = self.sequence_mask(
            sequence_length=length, max_len=y.size(1)
        ).unsqueeze(2)
        y_norm = self.sample_wise_min_max(y, mask)
        y_hat_norm = self.sample_wise_min_max(y_hat, mask)
        ssim_loss = self.loss_func(
            (y_norm * mask).unsqueeze(1), (y_hat_norm * mask).unsqueeze(1)
        )

        if ssim_loss.item() > 1.0:
            print(
                f" > SSIM loss is out-of-range {ssim_loss.item()}, setting it 1.0"
            )
            ssim_loss = torch.tensor(1.0, device=ssim_loss.device)

        if ssim_loss.item() < 0.0:
            print(
                f" > SSIM loss is out-of-range {ssim_loss.item()}, setting it 0.0"
            )
            ssim_loss = torch.tensor(0.0, device=ssim_loss.device)

        return ssim_loss


# Adopted from https://github.com/photosynthesis-team/piq
class _SSIMLoss(_Loss):
    """Creates a criterion that measures the structural similarity index error between
    each element in the input x and target y.
    Equation link: https://en.wikipedia.org/wiki/Structural_similarity
    x and y are tensors of arbitrary shapes with a total of n elements each.
    The sum operation still operates over all the elements, and divides by n.
    The division by n can be avoided if one sets reduction = sum.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.
    Arguments
    ---------
    kernel_size: int
        By default, the mean and covariance of a pixel is obtained
        by convolution with given filter_size.
    kernel_sigma: float
        Standard deviation for Gaussian kernel.
    k1: float
        Coefficient related to c1 (see equation in the link above).
    k2: float
        Coefficient related to c2 (see equation in the link above).
    downsample: bool
        Perform average pool before SSIM computation (Default: True).
    reduction: str
        Specifies the reduction type
    data_range: Union[int, float]
        Maximum value range of images (usually 1.0 or 255).
    Example
    -------
        >>> loss = _SSIMLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()
    """

    __constants__ = ["kernel_size", "k1", "k2", "sigma", "kernel", "reduction"]

    def __init__(
        self,
        kernel_size=11,
        kernel_sigma=1.5,
        k1=0.01,
        k2=0.03,
        downsample=True,
        reduction="mean",
        data_range=1.0,
    ):
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the ssim function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert (
            kernel_size % 2 == 1
        ), f"Kernel size must be odd, got [{kernel_size}]"
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.data_range = data_range

    def _reduce(self, x, reduction="mean"):
        """Reduce input in batch dimension if needed.
        Arguments
        ---------
        x: torch.Tensor
            Tensor with shape (B, *).
        reduction: str
            Specifies the reduction type:
            none | mean | sum (Default: mean)
        """
        if reduction == "none":
            return x
        if reduction == "mean":
            return x.mean(dim=0)
        if reduction == "sum":
            return x.sum(dim=0)
        raise ValueError(
            "Unknown reduction. Expected one of {'none', 'mean', 'sum'}"
        )

    def _validate_input(
        self,
        tensors,
        dim_range=(0, -1),
        data_range=(0.0, -1.0),
        size_range=None,
    ):
        """Check if the input satisfies the requirements
        Arguments
        ---------
        tensors: torch.Tensor
            Tensors to check
        dim_range: Tuple[int, int]
            Allowed number of dimensions. (min, max)
        data_range: Tuple[float, float]
            Allowed range of values in tensors. (min, max)
        size_range: Tuple[int, int]
            Dimensions to include in size comparison. (start_dim, end_dim + 1)
        """

        if not __debug__:
            return

        x = tensors[0]

        for t in tensors:
            assert torch.is_tensor(t), f"Expected torch.Tensor, got {type(t)}"
            assert (
                t.device == x.device
            ), f"Expected tensors to be on {x.device}, got {t.device}"

            if size_range is None:
                assert (
                    t.size() == x.size()
                ), f"Expected tensors with same size, got {t.size()} and {x.size()}"
            else:
                assert (
                    t.size()[size_range[0] : size_range[1]]
                    == x.size()[size_range[0] : size_range[1]]
                ), f"Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}"

            if dim_range[0] == dim_range[1]:
                assert (
                    t.dim() == dim_range[0]
                ), f"Expected number of dimensions to be {dim_range[0]}, got {t.dim()}"
            elif dim_range[0] < dim_range[1]:
                assert (
                    dim_range[0] <= t.dim() <= dim_range[1]
                ), f"Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}"

            if data_range[0] < data_range[1]:
                assert (
                    data_range[0] <= t.min()
                ), f"Expected values to be greater or equal to {data_range[0]}, got {t.min()}"
                assert (
                    t.max() <= data_range[1]
                ), f"Expected values to be lower or equal to {data_range[1]}, got {t.max()}"

    def gaussian_filter(self, kernel_size, sigma):
        """Returns 2D Gaussian kernel N(0,sigma^2)
        Arguments
        ---------
        size: int
            Size of the kernel
        sigma: float
            Std of the distribution
        Returns
        ---------
        gaussian_kernel: torch.Tensor
            [1, kernel_size, kernel_size]
        """
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= (kernel_size - 1) / 2.0

        g = coords ** 2
        g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

        g /= g.sum()
        return g.unsqueeze(0)

    def _ssim_per_channel(self, x, y, kernel, k1=0.01, k2=0.03):
        """Calculate Structural Similarity (SSIM) index for X and Y per channel.
        Arguments
        ---------
        x: torch.Tensor
            An input tensor (N, C, H, W).
        y: torch.Tensor
            A target tensor (N, C, H, W).
        kernel: torch.Tensor
            2D Gaussian kernel.
        k1: float
            Algorithm parameter (see equation in the link above).
        k2: float
            Algorithm parameter (see equation in the link above).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        Returns
        ---------
            Full Value of Structural Similarity (SSIM) index.
        """
        if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
            raise ValueError(
                f"Kernel size can't be greater than actual input size. Input size: {x.size()}. "
                f"Kernel size: {kernel.size()}"
            )

        c1 = k1 ** 2
        c2 = k2 ** 2
        n_channels = x.size(1)
        mu_x = F.conv2d(
            x, weight=kernel, stride=1, padding=0, groups=n_channels
        )
        mu_y = F.conv2d(
            y, weight=kernel, stride=1, padding=0, groups=n_channels
        )

        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_xx = (
            F.conv2d(
                x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu_xx
        )
        sigma_yy = (
            F.conv2d(
                y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu_yy
        )
        sigma_xy = (
            F.conv2d(
                x * y, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu_xy
        )

        # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
        cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

        # Structural similarity (SSIM)
        ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

        ssim_val = ss.mean(dim=(-1, -2))
        cs = cs.mean(dim=(-1, -2))
        return ssim_val, cs

    def _ssim_per_channel_complex(self, x, y, kernel, k1=0.01, k2=0.03):
        """Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.
        Arguments
        ---------
        x: torch.Tensor
            An input tensor (N, C, H, W, 2).
        y: torch.Tensor
            A target tensor (N, C, H, W, 2).
        kernel: torch.Tensor
            2-D gauss kernel.
        k1: float
            Algorithm parameter (see equation in the link above).
        k2: float
            Algorithm parameter (see equation in the link above).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        Returns:
            Full Value of Complex Structural Similarity (SSIM) index.
        """
        n_channels = x.size(1)
        if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
            raise ValueError(
                f"Kernel size can't be greater than actual input size. Input size: {x.size()}. "
                f"Kernel size: {kernel.size()}"
            )

        c1 = k1 ** 2
        c2 = k2 ** 2

        x_real = x[..., 0]
        x_imag = x[..., 1]
        y_real = y[..., 0]
        y_imag = y[..., 1]

        mu1_real = F.conv2d(
            x_real, weight=kernel, stride=1, padding=0, groups=n_channels
        )
        mu1_imag = F.conv2d(
            x_imag, weight=kernel, stride=1, padding=0, groups=n_channels
        )
        mu2_real = F.conv2d(
            y_real, weight=kernel, stride=1, padding=0, groups=n_channels
        )
        mu2_imag = F.conv2d(
            y_imag, weight=kernel, stride=1, padding=0, groups=n_channels
        )

        mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
        mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
        mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
        mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

        compensation = 1.0

        x_sq = x_real.pow(2) + x_imag.pow(2)
        y_sq = y_real.pow(2) + y_imag.pow(2)
        x_y_real = x_real * y_real - x_imag * y_imag
        x_y_imag = x_real * y_imag + x_imag * y_real

        sigma1_sq = (
            F.conv2d(
                x_sq, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                y_sq, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu2_sq
        )
        sigma12_real = (
            F.conv2d(
                x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu1_mu2_real
        )
        sigma12_imag = (
            F.conv2d(
                x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels
            )
            - mu1_mu2_imag
        )
        sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
        mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
        # Set alpha = beta = gamma = 1.
        cs_map = (sigma12 * 2 + c2 * compensation) / (
            sigma1_sq.unsqueeze(-1)
            + sigma2_sq.unsqueeze(-1)
            + c2 * compensation
        )
        ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (
            mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation
        )
        ssim_map = ssim_map * cs_map

        ssim_val = ssim_map.mean(dim=(-2, -3))
        cs = cs_map.mean(dim=(-2, -3))

        return ssim_val, cs

    def ssim(
        self,
        x,
        y,
        kernel_size=11,
        kernel_sigma=1.5,
        data_range=1.0,
        reduction="mean",
        full=False,
        downsample=True,
        k1=0.01,
        k2=0.03,
    ):
        """Interface of Structural Similarity (SSIM) index.
        Inputs supposed to be in range [0, data_range].
        To match performance with skimage and tensorflow set downsample = True.
        Arguments
        ---------
        x: torch.Tensor
            An input tensor (N, C, H, W) or (N, C, H, W, 2).
        y: torch.Tensor
            A target tensor (N, C, H, W) or (N, C, H, W, 2).
        kernel_size: int
            The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: float
            Sigma of normal distribution.
        data_range: Union[int, float]
            Maximum value range of images (usually 1.0 or 255).
        reduction: str
            Specifies the reduction type:
            none | mean | sum. Default:mean
        full: bool
            Return cs map or not.
        downsample: bool
            Perform average pool before SSIM computation. Default: True
        k1: float
            Algorithm parameter (see equation in the link above).
        k2: float
            Algorithm parameter (see equation in the link above).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        Returns
        ---------
            Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
            as a tensor of size 2.
        """
        assert (
            kernel_size % 2 == 1
        ), f"Kernel size must be odd, got [{kernel_size}]"
        self._validate_input(
            [x, y], dim_range=(4, 5), data_range=(0, data_range)
        )

        x = x / float(data_range)
        y = y / float(data_range)

        # Averagepool image if the size is large enough
        f = max(1, round(min(x.size()[-2:]) / 256))
        if (f > 1) and downsample:
            x = F.avg_pool2d(x, kernel_size=f)
            y = F.avg_pool2d(y, kernel_size=f)

        kernel = (
            self.gaussian_filter(kernel_size, kernel_sigma)
            .repeat(x.size(1), 1, 1, 1)
            .to(y)
        )
        _compute_ssim_per_channel = (
            self._ssim_per_channel_complex
            if x.dim() == 5
            else self._ssim_per_channel
        )
        ssim_map, cs_map = _compute_ssim_per_channel(
            x=x, y=y, kernel=kernel, k1=k1, k2=k2
        )
        ssim_val = ssim_map.mean(1)
        cs = cs_map.mean(1)

        ssim_val = self._reduce(ssim_val, reduction)
        cs = self._reduce(cs, reduction)

        if full:
            return [ssim_val, cs]

        return ssim_val

    def forward(self, x, y):
        """Computation of Structural Similarity (SSIM) index as a loss function.
        Arguments
        ---------
        x: torch.Tensor
            An input tensor (N, C, H, W) or (N, C, H, W, 2).
        y: torch.Tensor
            A target tensor (N, C, H, W) or (N, C, H, W, 2).
        Returns
        ---------
        Value of SSIM loss to be minimized, i.e 1 - ssim in [0, 1] range. In case of 5D input tensors,
        complex value is returned as a tensor of size 2.
        """

        score = self.ssim(
            x=x,
            y=y,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            downsample=self.downsample,
            data_range=self.data_range,
            reduction=self.reduction,
            full=False,
            k1=self.k1,
            k2=self.k2,
        )
        return torch.ones_like(score) - score
