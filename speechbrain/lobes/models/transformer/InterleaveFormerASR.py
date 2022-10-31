"""Transformer for ASR in the SpeechBrain sytle.
Authors
* Yiqi Wang, 2022
"""

import torch  # noqa 42
from torch import nn
from typing import Optional
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.InterleaveFormer import (
    InterleaveFormerInterface,
    get_lookahead_mask,
    get_lookahead_hopping_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)
from speechbrain.nnet.activations import Swish
from speechbrain.dataio.dataio import length_to_mask


class InterleaveFormerASR(InterleaveFormerInterface):
    """This is an implementation of InterleaveFormer model for ASR.
    The architecture is based on the paper "PLACE HODLER":
    arxiv PLACE HODLER
    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of causal-encoder-layers in the InterleaveFormer (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=0).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        InterleaveFormer as a causal encoder. No other option!
    conformer_activation: torch.nn.Module, optional
        NOT USED
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (InterleaveFormer is always causal).
        If causal the Conformer convolutional layer is causal.
    Example
    -------
    >>> src = torch.rand([8, 200, 512]) # 200 is the padded total length including many bi-modality segments
    >>> tgt = torch.randint(0, 720, [8, 200])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out = net.forward(src, tgt) # not that enc_out actually contains both audio and text
    >>> enc_out.shape
    torch.Size([8, 200, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=0,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "InterleaveFormer",
        conformer_activation: Optional[nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(self, src, tgt, wave_len, seg_stats = None, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wave_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        seg_stats:
            an array contains max len of src and max len of tgt
            # dictionary for milestones
            # The sum of each key's array is the total true length of a sequence where each element in the array indicates segment len
            # Format: { key_i: array for i in range(batch)} used by modality expert.
            # Key_i: key to index a sample's modality stats.
            # Value_of_key_i: an array. Even element is audio segment true length. Odd element is text tokens true length.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """
        # assert type(seg_stats) == type(dict()), f"Need seg_stats to be a valid dictionary for modality expert!"
        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask, # this one could be the hopping causal mask! Postponed right now.
        ) = self.make_masks(src, tgt, wave_len, seg_stats, pad_idx=pad_idx)

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None


        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            assert False, f"Don't support RelPosMHAXL yet"
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        # assert False, f"src shape: {src.shape} {src_key_padding_mask.shape} tgt {tgt.shape} {tgt_mask.shape}"
        
        final_src = torch.cat([src, tgt], dim = 1)
        final_padding = torch.cat([src_key_padding_mask, tgt_key_padding_mask], dim = 1)
        assert False, f"final src: {final_src.shape} padding: {final_padding.shape} causal hop: {tgt_mask.shape}"
        encoded_output, _ = self.encoder(
            src=src,
            seg_stats = seg_stats, # used by modality expert
            src_mask=tgt_mask, # this must be a causal mask, hopping style
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )


        # encoded_output is bi-modality learned representation.
        # None since we don't have a "decoder"
        return encoded_output, None

    def make_masks(self, src, tgt, wave_len = None, seg_stats = None, pad_idx=0):
        """This method generates the masks for training the transformer model.
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        wave_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        seg_stats:
            an array contains max len of src and max len of tgt
            # dictionary for milestones 2
            # The sum of each key's array is the total true length of a sequence where each element in the array indicates segment len
            # Format: { key_i: array for i in range(batch)} used by modality expert.
            # Key_i: key to index a sample's modality stats.
            # Value_of_key_i: an array. Even element is audio segment true length. Odd element is text tokens true length.
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if  wave_len is not None:
            # ??? HELP ME double check if below implementation make sense or not!
            # abs_len = [ sum(seg_stats[i]) for i in range(len(seg_stats))]
            abs_len = torch.round(wave_len * src.shape[1])
            src_key_padding_mask = ~length_to_mask(abs_len).bool()
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_hopping_mask(tgt, seg_stats) # hopping causal mask implemented in InterleaveFormer.py
        # tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

