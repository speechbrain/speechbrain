"""A simplistic Text-to-Speech model operating on
discrete/tokenized audio representations, available in both
Transformer and RNN flavours.

NOTE: This model does not use the standa
rd Transformer interface
in order to make it usable as both as a full model and as a
decoder-only model

Authors
* Artem Ploujnikov, 2023
"""

import math
from collections import namedtuple
from enum import Enum
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from speechbrain.dataio.dataio import clean_padding_, length_to_mask
from speechbrain.decoders.seq2seq import S2STransformerBeamSearcher
from speechbrain.lobes.models.transformer.Transformer import (
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerEncoder,
    get_lookahead_mask,
)
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.embedding import Embedding, MultiEmbedding
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from speechbrain.nnet.losses import compute_masked_loss, kldiv_loss, mse_loss
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.data_utils import concat_padded_features

TokotronOutput = namedtuple(
    "TokotronOutput",
    [
        "out",
        "gate_out",
        "p_eos",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
    ],
)

TokotronDecoderOutput = namedtuple(
    "TokotronDecoderOutput",
    ["out", "gate_out", "dec_self_attn", "dec_attn", "alignments", "context"],
)

TokotronDecoderInfernceOutput = namedtuple(
    "TokotronDecoderInferenceOutput",
    [
        "audio",
        "length",
        "dec_self_attn",
        "dec_attn",
        "alignments",
        "p_eos",
    ],
)

TokotronInfernceOutput = namedtuple(
    "TokotronInferenceOutput",
    [
        "audio",
        "length",
        "wav",
        "wav_length",
        "enc_self_attn",
        "dec_self_attn",
        "dec_attn",
        "alignments",
        "p_eos",
    ],
)

IGNORE_IN_STATE_DICT = {"vocoder"}


class EosMode(Enum):
    """Indicates how end-of-sequence will be determined

    EosMode.GATE : Using a separate gating mechanism similar to Tacotron2
    EosMode.TOKEN : Using a specialized token
    """

    GATE = "gate"
    TOKEN = "token"


class RepresentationMode(Enum):
    """Indicates the representation mode

    RepresentationMode.DISCRETE: Discrete representations (tokens)
    RepresentationMode.CONTINUOUS: Continuous representations (e.g. a single layer
        from an SSL model)"""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class TokotronTransformerDecoder(nn.Module):
    """The Tokotron decoder - can be used in a standalone model or as
    a component of a larger model

    Arguments
    ---------
    num_tokens : int, optional
        the number of tokens
    tokens_per_step : int, optional
        the number of tokens to be output, per transformer time step
    d_model : int, optional
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    attention_type : str
        The type of transformer attention to be used
    num_layers: int
        The number of layers
    dropout : float
        The dropout probability
    emb_dropout : float
        The embedding dropout probability
    target_dropout : float
        The target dropout probability
    audio_emb : torch.nn.Module, optional
        The audio embedding to be used
    audio_emb_size : int
        The size of the audio embeddings (if learned)
    activation : torch.nn.Module, optional
        The activation function to be used
    use_tgt_padding_mask : bool, optional
        whether to use a target padding mask
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
    max_decoder_steps : int, optional
        The maximum number of decoder steps used during training
    bos_width : int
        The number of time steps occupied by the beginning-of-sequence marker
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    audio_token_shift : int
        The number of positions audio tokens are shifted to "make room" for
        special tokens
    multihead_input : bool
        If set to true, targets are expected to be in the (Batch x Length x Heads x Dim)
        format. Otherwise, they are expected to be single-headed
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    audio_dim : int, optional
        The continuous audio input dimension
    emb : dict
        The embedding configuration
    max_audio_length : int
        The maximum length of the audio signal
    layerwise_renorm : bool
        Whether to apply normalization at the end of each decoder layer
    """

    def __init__(
        self,
        num_tokens=1024,
        tokens_per_step=2,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        attention_type="regularMHA",
        num_layers=6,
        dropout=0.2,
        emb_dropout=0.0,
        target_dropout=None,
        audio_emb=None,
        audio_emb_size=128,
        activation=nn.LeakyReLU,
        use_tgt_padding_mask=False,
        audio_emb_freeze=False,
        max_decoder_steps=1000,
        bos_width=1,
        gate_threshold=0.5,
        gate_offset=0,
        audio_token_shift=0,
        multihead_input=True,
        representation_mode=RepresentationMode.DISCRETE,
        audio_dim=1024,
        emb=None,
        max_audio_length=2000,
        layerwise_renorm=True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens_per_step = tokens_per_step
        self.dec = EmbeddingGuidedTransformerDecoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            emb_dropout=emb_dropout,
            emb=emb,
            layerwise_renorm=layerwise_renorm,
        )
        in_proj_size = audio_emb_size
        if multihead_input:
            in_proj_size *= tokens_per_step
        self.tgt_in_proj = Linear(
            input_size=in_proj_size,
            n_neurons=d_model,
        )
        self.representation_mode = RepresentationMode(representation_mode)
        self.out_dim = (
            num_tokens + audio_token_shift
            if self.representation_mode == RepresentationMode.DISCRETE
            else audio_dim
        )
        self.out_proj = Linear(
            input_size=d_model,
            n_neurons=self.out_dim * tokens_per_step,
        )
        self.gate = Linear(input_size=d_model, n_neurons=1)
        if audio_emb is None:
            if self.representation_mode == RepresentationMode.DISCRETE:
                audio_emb = MultiEmbedding(
                    num_embeddings=num_tokens + audio_token_shift,
                    embedding_dim=audio_emb_size,
                    num_heads=tokens_per_step,
                    normalized=True,
                    d_model=d_model,
                )
            else:
                audio_emb = Linear(
                    input_size=audio_dim,
                    n_neurons=audio_emb_size,
                )

        self.positional_encoding = PositionalEncoding(
            d_model,
            max_len=max_audio_length,
        )
        if target_dropout is None:
            target_dropout = dropout
        self.target_dropout = target_dropout
        self.audio_emb = audio_emb
        self.max_decoder_steps = max_decoder_steps
        self.attention_type = attention_type
        self.use_tgt_padding_mask = use_tgt_padding_mask
        self.audio_emb_freeze = audio_emb_freeze
        self.bos_width = bos_width
        self.gate_threshold = gate_threshold
        self.gate_offset = gate_offset
        if self.audio_emb_freeze:
            for parameter in self.audio_emb.parameters():
                parameter.requires_grad_(False)
        self.audio_token_shift = audio_token_shift
        self.multihead_input = multihead_input
        self.d_model = d_model
        self.d_model_sqrt = math.sqrt(d_model)

    def decode(
        self,
        enc_out,
        tgt,
        src_length=None,
        src_key_padding_mask=None,
        tgt_length=None,
        tgt_key_padding_mask=None,
        pos_embs_src=None,
        emb=None,
    ):
        """Performs a decode step

        Arguments
        ---------
        enc_out : torch.Tensor
            The raw encoder outputs
        tgt : torch.Tensor, optional
            Targets (i.e. the parts of the audio sequence that have
            already been decoded)
        src_length : torch.Tensor, optional
            Relative length of input sequences
        src_key_padding_mask : torch.Tensor, optional
            Key padding mask for the tensor (if pre-computed)
        tgt_length : torch.Tensor, optional
            The target relative length
        tgt_key_padding_mask : torch.Tensor, optional
            The target key padding mask (if pre-computed)
        pos_embs_src : torch.Tensor, optional
            The target positional embeddings
        emb : dict, optional
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)


        Returns
        -------
        dec_out : torch.Tensor
            Decoder outputs
        dec_self_attn : list
            Decoder self-attentions (list of tensors)
        dec_attn : list
            Decoder attention (list of tensors)
        """
        if src_length is not None and src_key_padding_mask is None:
            src_max_len = enc_out.size(1)
            src_key_padding_mask = length_to_mask(
                src_length * src_max_len, src_max_len
            ).logical_not()

        if (
            tgt_length is not None
            and tgt_key_padding_mask is None
            and self.use_tgt_padding_mask
        ):
            tgt_max_len = tgt.size(1)
            tgt_key_padding_mask = length_to_mask(
                tgt_length * tgt_max_len, tgt_max_len
            ).logical_not()

        audio_emb = self.audio_emb(tgt)
        if self.multihead_input:
            batch_size, audio_max_len, heads, audio_dim = audio_emb.shape
            audio_emb_combined = audio_emb.reshape(
                batch_size, audio_max_len, heads * audio_dim
            )
        else:
            audio_emb_combined = audio_emb
        tgt = self.tgt_in_proj(audio_emb_combined)
        tgt = F.dropout(tgt, self.target_dropout, training=self.training)

        if self.attention_type == "Mega" and self.chunk_size > 0:
            tgt_mask = get_lookahead_mask_for_length(
                self.chunk_size, device=tgt.device
            )
        else:
            tgt_mask = get_lookahead_mask(tgt)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_tgt = self.positional_encoding(tgt)
        elif self.attention_type == "Mega":
            pos_embs_tgt = None
        else:
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_tgt = None
        # NOTE: Normalization for continuous representations, similar
        # to NormalizedEmbedding
        if self.representation_mode == RepresentationMode.CONTINUOUS:
            tgt = tgt * self.d_model_sqrt
        (
            dec_out,
            dec_self_attn,
            dec_attn,
        ) = self.dec(
            tgt=tgt,
            memory=enc_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_tgt,
            pos_embs_src=pos_embs_src,
            emb=emb,
        )
        return dec_out, dec_self_attn, dec_attn

    def forward(
        self,
        enc_out,
        tgt,
        src_length=None,
        src_key_padding_mask=None,
        tgt_length=None,
        tgt_key_padding_mask=None,
        pos_embs_src=None,
        emb=None,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        enc_out : torch.Tensor
            Raw encoder outputs
        tgt : torch.Tensor
            Targets (audio tokens)
        src_length : torch.Tensor
            The relative lengths of the source sequence
        src_key_padding_mask : torch.Tensor
            Alternatively, the key padding mask for keys
        tgt_length : torch.Tensor
            Target lengths
        tgt_key_padding_mask : torch.Tensor
            Alternatively, the key padding mask for target keys
        pos_embs_src : dict
            Source positional embeddings
        emb : dict, optional
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)

        Returns
        -------
        result : TokotronDecoderOutput
            The model output

            out : torch.Tensor
                The outputs - token probabilities
                Batch x Length x Head x Token
            gate_out : torch.Tensor
                The gate activation tesnsor
                Batch x Length
            dec_self_attn : list
                Decoder self-attentions
            dec_attn : list
                Decoder multi-head attentions
            alignments : torch.Tensor
                Decoder multi-head attentions, concatenated
                as a single tensor
            context : dict
                An empty dictionary (not used in this decoder)
        """
        if self.representation_mode == RepresentationMode.DISCRETE:
            tgt_token_shift = torch.zeros(
                (1, tgt.size(1), 1), device=tgt.device
            )
            tgt_token_shift[:, self.bos_width :, :] += self.audio_token_shift
            tgt = tgt + tgt_token_shift
        dec_out, dec_self_attn, dec_attn = self.decode(
            enc_out,
            tgt,
            src_length,
            src_key_padding_mask,
            tgt_length,
            tgt_key_padding_mask,
            pos_embs_src,
            emb,
        )
        lin_out = self.out_proj(dec_out)
        batch_size, audio_max_len, num_tokens = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size,
            audio_max_len,
            self.tokens_per_step,
            num_tokens // self.tokens_per_step,
        )
        gate_out = self.gate(dec_out).squeeze(-1)
        return TokotronDecoderOutput(
            lin_out_heads,
            gate_out,
            dec_self_attn,
            dec_attn,
            get_alignments(dec_attn),
            {},
        )

    def init_audio_emb(self, emb):
        """Initializes audio embeddings with the specified embedding tensor - useful for re-using the
        embeddings from a pre-trained model

        Arguments
        ---------
        emb : torch.Tensor
            The embedding tensor with which to initialize
        """
        self.audio_emb.initialize(emb)


class TokotronTransformerAutoregressiveInference(nn.Module):
    """A greedy autoregressive inference implementation

    Arguments
    ---------
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    tokens_per_step : int
        The number of audio tokens per step (e.g. corresponding to quantizers
        in the downstream model)
    bos_idx : int, optional
        the Beginning-of-Sequence index
    max_steps : int, optional
        The maximum number of decoder steps used during training
    audio_token_shift : int, optional
        The number by which token indices will be shifted (used to introduce
        additional tokens)
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    audio_dim : int, optional
        The continuous audio input dimension
    show_inference_progress : bool, optional
        Whether to show inference progress in the console
    """

    def __init__(
        self,
        gate_offset,
        gate_threshold,
        tokens_per_step,
        bos_idx,
        max_steps,
        audio_token_shift,
        representation_mode=RepresentationMode.DISCRETE,
        audio_dim=1024,
        show_inference_progress=True,
    ):
        super().__init__()
        self.decoder = None
        self.gate_offset = gate_offset
        self.gate_threshold = gate_threshold
        self.tokens_per_step = tokens_per_step
        self.bos_idx = bos_idx
        self.max_steps = max_steps
        self.audio_token_shift = audio_token_shift
        self.representation_mode = RepresentationMode(representation_mode)
        self.audio_dim = audio_dim
        self.show_inference_progress = show_inference_progress

    def bind(self, model):
        """Binds this inference implementation to a model

        Arguments
        ---------
        model : TokotronTransformerModel
            The transformer model
        """
        self.decoder = model.decoder

    def forward(self, enc_out, length, emb=None):
        """Performs autoregressive inference

        Arguments
        ---------
        enc_out : torch.Tensor
            Raw encoder outputs

        length : torch.Tensor
            Relative lengths

        emb : dict, optional
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)

        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
        dec_self_attn : torch.Tensor
            Decoder self-attentions
        dec_attn : torch.Tensor
            Decoder multihead attentions (or equivalent)
        alignments : torch.Tensor
            Aggregated alignments from all layers
        p_eos : torch.Tensor
            End-of-sequence probability at each step
        """
        with torch.no_grad():
            gate_offset = int(round(self.gate_offset))
            batch_size = enc_out.size(0)

            # Initialize BOS
            bos = get_bos(
                batch_size,
                self.tokens_per_step,
                self.bos_idx,
                audio_dim=self.audio_dim,
                representation_mode=self.representation_mode,
                device=enc_out.device,
            )
            audio = bos
            audio_length = torch.ones(batch_size, device=enc_out.device)
            steps_range = range(self.max_steps)

            # Initialize the gate activation index
            seq_gate_idx = (
                torch.ones(batch_size, device=enc_out.device) * self.max_steps
            )

            # Initialize an indicator that tells whether the gate has activated
            # for a given sample
            seq_gate_act = torch.zeros(batch_size, device=enc_out.device).bool()

            # Show progress if enabled
            if self.show_inference_progress:
                steps_range = tqdm(steps_range, desc="Inference")
            for idx in steps_range:
                # One autoregressive step
                step_out = self.decoder.forward(
                    enc_out=enc_out,
                    src_length=length,
                    tgt=audio,
                    tgt_length=audio_length,
                    emb=emb,
                )
                audio_out = step_out.out

                if self.representation_mode == RepresentationMode.DISCRETE:
                    audio_out = audio_out.argmax(-1)

                # The model outputs predictions without BOS. Add the BOS back for the
                # following step

                audio = torch.cat([bos, audio_out], dim=1)

                # Find the gate activation of the current step
                step_gate_out = step_out.gate_out[:, -1]

                # Compute the gate activation (final sigmoid)
                step_gate_act = step_gate_out.sigmoid() > self.gate_threshold

                # Update the gate activation index as follows
                #
                # - If the gate has already activated in a previous step, leave the index as is
                # - Otherwise:
                #   - If the gate has activated in the current step, update it with the current
                #     step index
                #   - Otherwise, leave it as is
                seq_gate_idx = torch.where(
                    seq_gate_act,
                    seq_gate_idx,
                    torch.where(
                        step_gate_act,
                        torch.tensor(idx, device=step_gate_out.device),
                        seq_gate_idx,
                    ),
                )

                # Update the gate indicator
                seq_gate_act = seq_gate_act | step_gate_act

                # For a given sample, consider it done if the gate has activated at least
                # gate_offset steps ago
                seq_done = seq_gate_act & (idx - seq_gate_idx >= gate_offset)

                # Terminate inference if all samples are done
                done = seq_done.all()
                if done.item():
                    break

            # Length = gate activation index + the offset, not exceeding
            length_abs = (seq_gate_idx + gate_offset).clip(max=self.max_steps)
            max_inferred_len = length_abs.max().int()
            audio_out = audio_out[:, :max_inferred_len] - self.audio_token_shift
            # Compute relative lengths
            length = length_abs.float() / audio_out.size(1)

        if self.representation_mode == RepresentationMode.CONTINUOUS:
            audio_out = bipolar_compression_inv(audio_out)

        return TokotronDecoderInfernceOutput(
            audio=audio_out,
            length=length,
            dec_self_attn=step_out.dec_self_attn,
            dec_attn=step_out.dec_attn,
            alignments=step_out.alignments,
            p_eos=step_out.gate_out.sigmoid(),
        )


class TokotronSearchWrapper(nn.Module):
    """A wrapper class to facilitate seach-based inference. It takes care of re-interpreting
    a multi-headed sequence as multiple samples, for compatibility, and for the retention
    of attention tensors

    Arguments
    ---------
    decoder : TokotronTransformerDecoder
        the Tokotron transformer decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.tokens_per_step = decoder.tokens_per_step
        self.decoder = decoder

    def decode(self, memory, enc_states, enc_lens):
        """Wraps the decode operation, will all the necessary
        reshaping

        Arguments
        ---------
        memory : torch.Tensor
            Characters predicted so far
        enc_states : torch.Tensor
            Encoder states
        enc_lens : torch.Tensor
            Encoder state lengths

        Returns
        -------
        dec_out : torch.Tensor
            Decoder outputs
        dec_attn : torch.Tensor
            Decoder attentions
        """
        batch_size = enc_states.size(0) // self.tokens_per_step
        _, mem_len = memory.shape
        memory = memory.reshape(
            self.tokens_per_step, batch_size, mem_len
        ).permute(1, 2, 0)
        dec_out, dec_self_attn, dec_attn = self.decoder.decode(
            enc_out=enc_states[:batch_size],
            src_length=enc_lens[:batch_size],
            tgt=memory,
        )
        self.dec_self_attn = dec_self_attn
        self.dec_attn = dec_attn
        return dec_out, dec_attn


class TokotronTransformerBeamSearcher(S2STransformerBeamSearcher):
    """A slight modification of S2STransformerBeamSearcher that uses an
    explicit number of tokens instead of trying to infer it from the
    weights of the linear layer. This is needed because Tokotron is
    multi-header and the final output layer outputs multiple output states

    Arguments
    ---------
    num_tokens : int
        The number of audio tokens available
    *args : list
        Additional arguments (passed through)
    **kwargs : dict
        Additional keyword arguments (passed through)
    """

    def __init__(self, num_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens = num_tokens

    def set_n_out(self):
        """Set the number of output tokens."""
        return self.num_tokens


class SearchLinearWrapper(nn.Module):
    """A wrapper for the final linear layer of the Transformer. The goal is to
    make it compatible with the SpeechBrain Beam Search implementation, which is
    single-headed, by expanding multiple heads along the batch dimensions.

    Arguments
    ---------
    lin : torch.Tensor
        A linear layer with an output feature dimensions of
        (tokens_per_step x num_tokens)
    tokens_per_step : int
        the number of tokens the model outputs for each
        time step
    """

    def __init__(self, lin, tokens_per_step):
        super().__init__()
        self.lin = lin
        self.tokens_per_step = tokens_per_step

    def forward(self, x):
        """Performs a forward pass with all the required reshape operations

        Arguments
        ---------
        x : torch.Tensor
            The decoder output

        Returns
        -------
        result : torch.Tensor
            The layer output, reshaped along the batch dimension
        """
        x = self.lin(x)
        batch_size, max_len, out_dim = x.shape
        num_tokens = x.size(-1) // self.tokens_per_step
        x = (
            # batch x tokens x length
            x.transpose(2, 1)
            # batch x heads x tokens x length
            .view(batch_size, self.tokens_per_step, num_tokens, max_len)
            # heads x batch x tokens x length
            .transpose(0, 1)
            # heads * batch x tokens x length
            .reshape(self.tokens_per_step * batch_size, num_tokens, max_len)
            # heads * batch x length x tokens
            .transpose(1, 2)
        )
        return x


class TokotronTransformerModel(nn.Module):
    """An end-to-end Tokotron model receiving characters or phonemes
    as inputs and outputting audio tokens

    Arguments
    ---------
    input_num_tokens : int
        The number of input characters or phonemes available
    audio_num_tokens : int, optional
        The number of audio tokens
    audio_tokens_per_step : int, optional
        The number of output audio tokens per transformer step.
        When using Vocodec, this corresponds to the number of
        quantizers in the model used
    d_model : int, optional
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    attention_type : str, optional
        The type of attention to be used
    enc_num_layers : int, optional
        The number of encoder layers in1ì the encoder.
    dec_num_layers : int, optional
        The number of decoder layers in the decoder.
    dropout : int, optional
        The dropout value.
    target_dropout : float, optional
        The dropout probability for targets
    emb_dropout : float, optional
        The embedding dropout. Turned off by default (set to 0), can be
        useful in scenarios where embeddings are injected
    activation : torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    max_audio_length : int
        The maximum number of tokens to be output
    infer_max_audio_length : int
        The maximum number of tokens to be output, during inference
    bos_idx : int, optional
        the Beginning-of-Sequence index
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    use_tgt_padding_mask : bool, optional
        Whether to use a target padding mask
    audio_emb_size : int
        The size of the audio embedding
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
    show_inference_progress : bool, optional
        Whether to show inference progress in the console
    vocoder : nn.Module
        The vocoder module
    eos_mode : EosMode | str, optional
        the way the end of sequence is computed
    inference : TokotronInference, optional
        the inference method to be used
    audio_token_shift : int, optional
        The number by which token indices will be shifted (used to introduce
        additional tokens)
    scale_factor : float, optional
        forward decoding only - the scaling factor for
        targets in non-autoregressive inference
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    audio_dim : int, optional
        The continuous audio input dimension
    emb : dict, optional
        Available embeddings

        Example:
        {
            "spk": {
                "kind": "pretrained"
                "dim" : 512
            },
            "lang": {
                "kind": "trained",
                "count" : 2
            }
        }
    layerwise_renorm : bool
        Whether to apply normalization at the end of each decoder layer

    Example
    -------
    >>> import torch
    >>> emb_cfg = {
    ...     "spk": {
    ...         "kind": "pretrained",
    ...         "dim": 192,
    ...         "injection": "add"
    ...     }
    ... }
    >>> model = TokotronTransformerModel(input_num_tokens=26, emb=emb_cfg)
    >>> src = torch.randint(0, 26, (2, 50))
    >>> src_length = torch.ones(2)
    >>> tgt = torch.randint(0, 1024, (2, 100, 2))
    >>> tgt_length = torch.ones(2)
    >>> spk_emb = torch.randn(2, 192)
    >>> emb = {"spk": spk_emb}
    >>> tok_out = model(
    ...     input_tokens=src,
    ...     input_length=src_length,
    ...     audio=tgt,
    ...     audio_length=tgt_length,
    ...     emb=emb
    ... )
    >>> tok_out.out.shape
    torch.Size([2, 100, 2, 1024])
    >>> tok_out.alignments.shape
    torch.Size([2, 100, 50])
    >>> tok_out.p_eos.shape
    torch.Size([2, 100])
    """

    def __init__(
        self,
        input_num_tokens,
        audio_num_tokens=1024,
        audio_tokens_per_step=2,
        d_model=512,
        d_ffn=2048,
        nhead=4,
        attention_type="regularMHA",
        enc_num_layers=6,
        dec_num_layers=6,
        dropout=0.2,
        target_dropout=0.2,
        emb_dropout=0.0,
        activation=nn.LeakyReLU,
        max_audio_length=1000,
        infer_max_audio_length=None,
        bos_idx=0,
        gate_threshold=0.5,
        gate_offset=0,
        use_tgt_padding_mask=False,
        audio_emb_size=128,
        audio_emb_freeze=False,
        show_inference_progress=True,
        vocoder=None,
        eos_mode=EosMode.GATE,
        inference=None,
        audio_token_shift=0,
        scale_factor=5.0,
        representation_mode=RepresentationMode.DISCRETE,
        audio_dim=1024,
        emb=None,
        layerwise_renorm=True,
    ):
        super().__init__()
        self.in_emb = Embedding(
            num_embeddings=input_num_tokens,
            embedding_dim=d_model,
        )
        self.eos_mode = EosMode(eos_mode)
        self.d_model = d_model
        self.audio_token_shift = 1 if eos_mode == EosMode.TOKEN else 0
        self.encoder = TransformerEncoder(
            num_layers=enc_num_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            dropout=dropout,
            activation=activation,
            normalize_before=True,
        )
        audio_emb = None
        self.decoder = TokotronTransformerDecoder(
            num_tokens=audio_num_tokens + self.audio_token_shift,
            tokens_per_step=audio_tokens_per_step,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=dec_num_layers,
            activation=activation,
            dropout=dropout,
            target_dropout=target_dropout,
            emb_dropout=emb_dropout,
            use_tgt_padding_mask=use_tgt_padding_mask,
            audio_emb=audio_emb,
            audio_emb_size=audio_emb_size,
            audio_emb_freeze=audio_emb_freeze,
            max_decoder_steps=max_audio_length,
            gate_threshold=gate_threshold,
            gate_offset=gate_offset,
            audio_token_shift=audio_token_shift,
            multihead_input=True,
            representation_mode=representation_mode,
            audio_dim=audio_dim,
            emb=emb,
            layerwise_renorm=layerwise_renorm,
            max_audio_length=max_audio_length,
        )
        self.bos_idx = bos_idx
        self.vocoder = vocoder
        self.attention_type = attention_type
        self.gate_offset = gate_offset
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_len=max_audio_length
            )

        if inference is None:
            inference = TokotronTransformerAutoregressiveInference(
                gate_offset=self.gate_offset,
                gate_threshold=gate_threshold,
                tokens_per_step=audio_tokens_per_step,
                bos_idx=bos_idx,
                max_steps=infer_max_audio_length,
                audio_token_shift=self.audio_token_shift,
                representation_mode=representation_mode,
                audio_dim=audio_dim,
                show_inference_progress=show_inference_progress,
            )
        elif callable(inference) and not isinstance(inference, nn.Module):
            inference = inference()
        self.inference = inference
        self.inference.bind(self)
        self.scale_factor = scale_factor
        self.representation_mode = RepresentationMode(representation_mode)
        self.audio_dim = audio_dim
        if emb is not None:
            self.emb_proj = self._build_emb_proj(emb)
            self.vocoder_emb = [
                key
                for key, emb_config in emb.items()
                if emb_config.get("vocoder")
            ]
        self.emb_dropout = emb_dropout

    def _build_emb_proj(self, emb):
        """Builds the embedding projection

        Arguments
        ---------
        emb : dict
            Embedding configuration

        Returns
        -------
        emb_proj : torch.nn.ModuleDict
            embedding projections for each embedding"""
        emb_proj = {}
        for key, emb_config in emb.items():
            kind = emb_config.get("kind", "learned")
            if kind == "pretrained":
                emb_mod = Linear(
                    input_size=emb_config.get("dim", self.d_model)
                    + self.d_model,
                    n_neurons=self.d_model,
                )
            elif kind == "learned":
                emb_count = emb_config["count"]
                emb_dim = emb_config.get("dim", self.d_model)
                emb_mod = Embedding(
                    num_embeddings=emb_count, embedding_dim=emb_dim
                )
            else:
                raise ValueError(f"Invallid embedding kind: {kind}")
            emb_proj[key] = emb_mod
        return nn.ModuleDict(emb_proj)

    def __setattr__(self, name, value):
        """Prevents the vocoder from being saved in state_dict() - it is not typically fine-tuned
        and fine-tuning it would not be trivial

        Arguments
        ---------
        name : str
            The attribute name
        value : any
            The attribute value
        """
        if name in IGNORE_IN_STATE_DICT:
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        Arguments
        ---------
        state_dict : dict
            A dict containing parameters and persistent buffers.
        strict : bool, optional
            Whether to strictly enforce that the keys
        assign : bool, optional
            Whether to assign items in the state
            dictionary to their corresponding keys in the module

        Returns
        -------
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
        """
        state_dict = _filter_state_dict(state_dict)
        try:
            return super().load_state_dict(
                state_dict, strict=strict, assign=assign
            )
        except TypeError:
            # NOTE: Older versions of PyTorch don't have the assign parameter
            return super().load_state_dict(state_dict, strict=strict)

    @property
    def gate_offset(self):
        """The number of steps following gate activation to include"""
        return self.decoder.gate_offset

    @gate_offset.setter
    def gate_offset(self, value):
        """The number of steps following gate activation to include"""
        self.decoder.gate_offset = value

    def forward(
        self, input_tokens, input_length, audio, audio_length, emb=None
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        audio : torch.Tensor
            a (Batch x Length) tensor of output audio tokens (e.g. encodec)
        audio_length : torch.Tensor
            a 1-D tensor of relative output lengths
        emb : dict
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)

        Returns
        -------
        result : TokotronOutput
            Forward step outputs
            out : torch.Tensor
                The outputs - token probabilities
                Batch x Length x Head x Token
            gate_out : torch.Tensor
                The gate activation tesnsor
                Batch x Length
            dec_self_attn : list
                Encoder self-attentions
            dec_self_attn : list
                Decoder self-attentions
            dec_attn : list
                Decoder multi-head attentions
            alignments : torch.Tensor
                Decoder multi-head attentions, concatenated
                as a single tensor
        """

        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )
        if self.representation_mode == RepresentationMode.CONTINUOUS:
            audio = bipolar_compression(audio)

        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        tgt = audio
        tgt_length = audio_length
        enc_out = self.add_emb(enc_out, emb)
        dec_out = self.decoder(
            enc_out=enc_out,
            tgt=tgt,
            tgt_length=tgt_length,
            src_length=input_length,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs_src=pos_embs_encoder,
            emb=emb,
        )
        return TokotronOutput(
            out=dec_out.out,
            gate_out=dec_out.gate_out,
            p_eos=dec_out.gate_out.sigmoid(),
            enc_self_attn=enc_self_attn,
            dec_self_attn=dec_out.dec_self_attn,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
        )

    def add_emb(self, src, emb):
        """Adds embedding projections to the source tensor

        Arguments
        ---------
        src : torch.Tensor
            The source tensor
        emb : dict
            The embedding dictionary

        Returns
        -------
        result : torch.Tensor
            The resulting tensor, with embeddings incorporated
        """
        result = src
        if emb is not None:
            for key, emb_t in emb.items():
                batch_size, seq_len, feat_size = src.shape
                emb_size = emb_t.size(-1)
                emb_t_norm = nn.functional.layer_norm(emb_t, emb_t.shape)
                emb_exp = emb_t_norm.unsqueeze(1).expand(
                    batch_size, seq_len, emb_size
                )
                emb_exp = nn.functional.dropout(
                    emb_exp,
                    self.emb_dropout,
                )
                src_norm = nn.functional.layer_norm(src, src.shape)
                src_with_emb = torch.cat([src_norm, emb_exp], dim=-1)
                result = self.emb_proj[key](src_with_emb)
                src = result
        return result

    def process_inputs(self, input_tokens, input_length):
        """Computes embeddings, the padding mask and encoder
        positional embeddings

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

        Returns
        -------
        src : torch.Tensor
            input embeddings
        src_key_padding_mask : torch.Trnsor
            the key padding mask for inputs
        pos_emb_encoder : torch.Tensor
            encoder positional embeddings
        """
        in_emb = self.in_emb(input_tokens)
        pos_embs_encoder = None
        if self.attention_type == "RelPosMHAXL":
            src = in_emb
            pos_embs_encoder = self.positional_encoding(in_emb)
        else:
            src = in_emb + self.positional_encoding(
                in_emb
            )  # add the encodings here
            pos_embs_encoder = None

        input_max_len = input_tokens.size(1)
        src_key_padding_mask = length_to_mask(
            input_length * input_max_len,
            input_max_len,
        ).logical_not()
        return src, src_key_padding_mask, pos_embs_encoder

    def infer(self, input_tokens, input_length, emb=None):
        """Performs end-to-end inference

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        emb : dict, optional
            a [str, tensor] dictionary of embeddings (e.g. speaker, language,
            etc)


        Returns
        -------
        audio_tokens : torch.Tensor
            A (Batch x Length x Tokens) tensor of audio tokens
        length : torch.Tensor
            Inferred relative lengths
        wav : torch.Tensor
            Synthesized waveforms, if a vocoder is provided
        wav_length : torch.Tensor
            Waveform lengths
        enc_self_attn : torch.Tensor
            Encoder self-attentions
        dec_self_attn : torch.Tensor
            Decoder self-attentions
        dec_attn : torch.Tensor
            Decoder multihead attentions (or equivalent)
        alignments : torch.Tensor
            Aggregated alignments
        p_eos : torch.Tensor
            End-of-sequence probability at each step

        """
        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )
        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        enc_out = self.add_emb(enc_out, emb)
        dec_out = self.inference(enc_out, input_length, emb=emb)
        audio, audio_length = dec_out.audio, dec_out.length
        if self.vocoder is not None:
            if emb is None:
                vocoder_emb = {}
            else:
                vocoder_emb = {
                    key: value
                    for key, value in emb.items()
                    if key in self.vocoder_emb
                }
            wav = self.vocoder(audio, **vocoder_emb)
            clean_padding_(wav, dec_out.length)
        return TokotronInfernceOutput(
            audio=audio,
            length=audio_length,
            wav=wav,
            wav_length=dec_out.length,
            enc_self_attn=enc_self_attn,
            dec_self_attn=dec_out.dec_self_attn,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
            p_eos=dec_out.p_eos,
        )

    def init_audio_emb(self, emb):
        """Initializes audio embeddings with the specified embedding tensor - useful for re-using the
        embeddings from a pre-trained model

        Arguments
        ---------
        emb : torch.Tensor
            The embedding tensor with which to initialize
        """
        self.decoder.init_audio_emb(emb)


def get_bos(
    batch_size,
    tokens_per_step,
    bos,
    audio_dim=None,
    representation_mode=RepresentationMode.DISCRETE,
    device="cpu",
):
    """Constructs a beginning-of-sequence (BOS) sequence for
    autoregressive inference

    Arguments
    ---------
    batch_size : int
        The size of the batch dimension
    tokens_per_step : int
        The number of tokens per step
    bos : int | float
        The BOS token index for discrete representations or the
        value to be used for continuous representations
    audio_dim : int
        The dimension of audio representations (used if representation_mode is set to
        CONTINUOUS)
    representation_mode : RepresentationMode | str, optional
        the type of representations to be used (discrete or continuous)
    device : str|torch.Device
        The device identifier

    Returns
    -------
    seq: torch.Tensor
        the sequence consisting only of BOS"""
    if representation_mode == RepresentationMode.DISCRETE:
        seq = torch.ones(batch_size, 1, tokens_per_step, device=device) * bos
    else:
        seq = (
            torch.ones(batch_size, 1, tokens_per_step, audio_dim, device=device)
            * bos
        )
    return seq


def get_gate_targets(lengths, out_len):
    """Computes gate tarets and weights for each position

    Arguments
    ---------
    lengths : torch.Tensor
        Relative lengths
    out_len: int
        The maximum output length

    Returns
    -------
    targets : torch.Tensor
        Targets for gate outputs - EOS positions are marked as 1,
        non-EOS positions are marked at 0
    weights : torch.Tensor
        Weights by which individual position losses will be multiplied
    """
    pos = torch.arange(out_len, device=lengths.device)[None, :]
    gate_targets = pos >= (lengths * out_len)[:, None]
    gate_weights = torch.where(
        gate_targets,
        0.5 / (1.0 - lengths)[:, None],
        0.5 / lengths[:, None],
    )
    return gate_targets.float(), gate_weights


def get_alignments(attn):
    """Aggregates alignments from multiple layers and heads

    Arguments
    ---------
    attn: list
        raw attentions returned from a Transformer

    Returns
    -------
    alignments: torch.Tensor
        The resulting alignments
    """
    return torch.cat([item.unsqueeze(-1) for item in attn], dim=-1).mean(dim=-1)


TokotronLossDetails = namedtuple(
    "TokotronLossDetails",
    ["loss", "seq_loss", "gate_loss", "attn_loss"],
)


class TokotronLoss(nn.Module):
    """The loss module for the Tokotron module, combining
    a sequence loss a guided attention loss and a gate loss
    for end-of-sequence prediction

    Arguments
    ---------
    guided_attention_weight : float
        The relative weight of the guided attention loss
    guided_attention_sigma : float
        The sigma hyperparameter for the guided attention loss
        A higher sigma means a lower penalties for attention off
        the diagonal
    gate_weight : float
        The weight of the gate loss
    gate_beta : float
        The beta parameter for the distance difference loss
        used for the EOS gate

        See speechbrain.nnet.losses.distance_diff_loss
        - the beta parameter
    gate_gamma : float
        The gamma parameter for the distance difference loss
        used for the EOS gate

        See speechbrain.nnet.losses.distance_diff_loss
        - the gamma parameter
    gate_max_weight : float
        The maximum distance difference loss weight

        See speechbrain.nnet.losses.distance_diff_loss
        - the max_weight parameter
    silence_padding : float
        The amount of silence padding added to sequences
    seq_cost : float
        The type of sequence loss to be used
    eos_mode : EosMode
        The method of determining where the end of the sequence
        occurs

        EosMode.GATE: Use a separate "gate" projection
        EosMode.TOKEN: an EOS token
    audio_token_shift : int
        The number of positions audio tokens are shifted to accommodate special tokens
    bos_width : int
        The number of time steps occupied by the beginning-of-sequence marker
    eos_index : int
        The index of the end-of-sequence marker
    eos_width : int
        The number of time steps occupied by the end-of-sequence marker
    audio_tokens_per_step : int
        The number of audio tokens per step
    representation_mode : RepresentationMode
        the type of representations being used (discrete or continuous)
    """

    def __init__(
        self,
        guided_attention_weight,
        guided_attention_sigma,
        gate_weight,
        gate_beta,
        gate_gamma,
        gate_max_weight=1.0,
        silence_padding=0,
        seq_cost=None,
        eos_mode=EosMode.GATE,
        audio_token_shift=0,
        bos_width=1,
        eos_index=0,
        eos_width=1,
        audio_tokens_per_step=1,
        representation_mode=RepresentationMode.DISCRETE,
    ):
        super().__init__()
        self.guided_attention_weight = guided_attention_weight
        self.gate_weight = gate_weight
        self.gate_beta = gate_beta
        self.gate_gamma = gate_gamma
        self.gate_max_weight = gate_max_weight
        self.silence_padding = silence_padding
        self.representation_mode = RepresentationMode(representation_mode)
        if seq_cost is None:
            seq_cost = (
                kldiv_loss
                if self.representation_mode == RepresentationMode.DISCRETE
                else mse_loss
            )
        self.seq_cost = seq_cost
        self.attn_cost = GuidedAttentionLoss(
            sigma=guided_attention_sigma,
        )
        self.eos_mode = EosMode(eos_mode)
        self.audio_token_shift = audio_token_shift
        self.bos_width = bos_width
        self.eos_index = eos_index
        self.eos_width = eos_width
        if self.eos_mode == EosMode.TOKEN:
            audio_eos = (
                torch.ones(eos_width, audio_tokens_per_step).long() * eos_index
            )
            self.register_buffer("audio_eos", audio_eos)

    def forward(
        self,
        predictions,
        audio,
        audio_length,
        input_tokens,
        input_length,
        reduction="mean",
    ):
        """Computes the loss, with details

        Arguments
        ---------
        predictions : TokotronOutput
            the raw predictions, from the model
        audio : torch.Tensor
            target audio tokens
        audio_length : torch.Tensor
            relative lengths of target audio, for masking
        input_tokens : torch.Tensor
            input tokens (text of phonemes)
        input_length : torch.Tensor
            relative lengths of input tokens, for masking
        reduction : str
            loss reduction (see speechbrain.nnet.losses)

        Returns
        -------
        loss : torch.Tensor
            The total weighted loss
        seq_loss : torch.Tensor
            The sequence loss on the audio token sequence
        gate_loss : torch.Tensor
            The gate loss (if applicable)
        attn_loss : torch.Tensor
            The guided attention loss
        """
        out = predictions.out
        if self.representation_mode == RepresentationMode.DISCRETE:
            out = out.log_softmax(dim=-1)
        batch_size, out_len, heads, tok_dim = out.shape
        max_len = out_len - self.bos_width
        out_reshaped = (
            out.transpose(1, 2).reshape(batch_size * heads, out_len, tok_dim)
        )[:, :max_len]
        if self.eos_mode == EosMode.TOKEN:
            # NOTE: Shift only the tokens, but not EOS
            padding_lengths = torch.ones(batch_size, device=audio.device)
            audio_eos = self.audio_eos.unsqueeze(0).expand(
                batch_size, self.eos_width, heads
            )
            features = [audio + self.audio_token_shift, audio_eos]
            features = [item.float() for item in features]
            audio, audio_length = concat_padded_features(
                features,
                [audio_length, padding_lengths],
                dim=1,
            )

        tok_len = audio.size(1)
        if self.representation_mode == RepresentationMode.DISCRETE:
            audio_reshaped = audio.transpose(1, 2).reshape(
                batch_size * heads, max_len
            )
        else:
            audio_dim = audio.size(-1)
            audio_reshaped = audio.transpose(1, 2).reshape(
                batch_size * heads, max_len, audio_dim
            )
            audio_reshaped = bipolar_compression(audio_reshaped)
        audio_reshaped = audio_reshaped[:, :max_len]
        lengths_reshaped = (
            audio_length.unsqueeze(-1)
            .expand(batch_size, heads)
            .reshape(batch_size * heads)
        )
        seq_loss = self.seq_cost(
            out_reshaped[:, :tok_len],
            audio_reshaped,
            length=lengths_reshaped,
            reduction=reduction,
        )
        if reduction == "batch":
            seq_loss = seq_loss.reshape(batch_size, heads).mean(-1)
        lengths_abs = audio_length * out_len

        attn_loss = self.attn_cost(
            predictions.alignments,
            input_lengths=input_length * input_tokens.size(1),
            target_lengths=lengths_abs,
            reduction=reduction,
        )
        if self.eos_mode == EosMode.GATE:
            # NOTE: This adjustment will allow the gate to be "off" by up to silence_padding,
            # resulting in extra silence being output
            gate_loss = distance_diff_loss(
                predictions.p_eos,
                lengths_abs - self.silence_padding,
                beta=self.gate_beta,
                gamma=self.gate_gamma,
                max_weight=self.gate_max_weight,
                two_sided=True,
                reduction=reduction,
            )
        else:
            gate_loss = self._zero_loss(
                batch_size=batch_size,
                reduction=reduction,
                device=predictions.out.device,
            )

        loss = (
            seq_loss
            + self.guided_attention_weight * attn_loss
            + self.gate_weight * gate_loss
        )
        return TokotronLossDetails(loss, seq_loss, gate_loss, attn_loss)

    def _zero_loss(self, batch_size, reduction, device):
        if reduction == "batch":
            loss = torch.zeros((batch_size,), device=device)
        else:
            loss = torch.tensor(0.0, device=device)
        return loss


def _filter_state_dict(state_dict):
    """Removes ignored keys from state_dict.

    Arguments
    ---------
    state_dict : dict
        the raw state_dict

    Returns
    -------
    result : dict
        the filtered state_dict
    """
    return {
        key: value
        for key, value in state_dict.items()
        if not any(
            key.startswith(ignored_key + ".")
            for ignored_key in IGNORE_IN_STATE_DICT
        )
    }


def distance_diff_loss(
    predictions,
    targets,
    length=None,
    beta=0.25,
    max_weight=100.0,
    gamma=1.0,
    two_sided=False,
    reduction="mean",
):
    """A loss function that can be used in cases where a model outputs
    an arbitrary probability distribution for a discrete variable on
    an interval scale, such as the length of a sequence, and the ground
    truth is the precise values of the variable from a data sample.

    The loss is defined as
    loss_i = p_i * (exp(beta * |i - y|) - 1.) * gamma

    The loss can also be used where outputs aren't probabilities, so long
    as high values close to the ground truth position and low values away
    from it are desired

    Arguments
    ---------
    predictions : torch.Tensor
        a (batch x max_len) tensor in which each element is a probability,
        weight or some other value at that position

    targets : torch.Tensor
        a 1-D tensor in which each elemnent is thr ground truth

    length : torch.Tensor
        lengths (for masking in padded batches)

    beta : float
        a hyperparameter controlling the penalties, an exponent multiplier.
        With a higher beta, penalties will increase faster

    max_weight: torch.Tensor
        the maximum distance weight (for numerical stability in long sequences)

    gamma : float
        a global multiplier - used control the shape of the weighting function

    two_sided : bool
        if set to true, a penalty is added for outputting a low probability
        close to the end

    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size

    Returns
    -------
    loss : torch.Tensor
        The computed loss value

    Example
    -------
    >>> predictions = torch.tensor(
    ...    [[0.25, 0.5, 0.25, 0.0],
    ...     [0.05, 0.05, 0.9, 0.0],
    ...     [8.0, 0.10, 0.05, 0.05]]
    ... )
    >>> targets = torch.tensor([2., 3., 1.])
    >>> length = torch.tensor([.75, .75, 1.])
    >>> loss = distance_diff_loss(predictions, targets, length)
    >>> loss
    tensor(0.2967)
    """
    return compute_masked_loss(
        partial(
            _distance_diff_loss,
            beta=beta,
            max_weight=max_weight,
            two_sided=two_sided,
            gamma=gamma,
        ),
        predictions=predictions,
        targets=targets,
        length=length,
        reduction=reduction,
        mask_shape="loss",
    )


def distance_diff_loss_ramp(beta, max_weight, gamma):
    """For distance_diff_loss, calculates the number of steps from the ground truth
    at which the weight reaches the maximum

    Arguments
    ---------
    beta : float
        A hyperparameter controlling the penalties. With a higher beta,
        penalties will increase faster
    max_weight : torch.Tensor
        The maximum distance loss weight
    gamma : float
        A global linear multiplier - used control the shape of the weighting
        function

    Returns
    -------
    loss : torch.Tensor
        The loss value
    """
    return math.log(max_weight / gamma - 1) / beta


def _distance_diff_loss(
    predictions, targets, beta, max_weight, gamma, two_sided=False
):
    """Computes the raw (unreduced) distance difference loss

    Arguments
    ---------
    predictions: torch.Tensor
        a (batch x max_len) tensor in which each element is a probability,
        weight or some other value at that position

    targets: torch.Tensor
        a 1-D tensor in which each elemnent is thr ground truth

    beta: torch.Tensor
        a hyperparameter controlling the penalties. With a higher beta,
        penalties will increase faster

    max_weight: torch.Tensor
        the maximum distance weight (for numerical stability in long sequences)

    gamma : float
        a global multiplier - used control the shape of the weighting function

    two_sided : bool
        if set to true, a penalty is added for outputting a low probability
        close to the end

    Returns
    -------
    loss : torch.Tensor
        The loss value
    """
    batch_size, max_len = predictions.shape
    pos_range = (torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)).to(
        predictions.device
    )
    diff_range = (pos_range - targets.unsqueeze(-1)).abs()
    loss_weights = (((beta * diff_range).exp() - 1.0) * gamma).clamp(
        max=max_weight
    )
    loss = loss_weights * predictions
    if two_sided:
        flip_loss = (max_weight - loss_weights) * (1 - predictions)
        loss = loss + flip_loss
    return loss


def bipolar_compression(x):
    """The bipolar compression function
    f(x) = sign(x) ln(|x| + 1)
    """
    return x.sign() * (x.abs() + 1).log()


def bipolar_compression_inv(x):
    """The inverse of bipolar_compression"""
    return torch.where(x >= 0, x.exp() - 1.0, 1.0 - (-x).exp())


class EmbeddingGuidedTransformerDecoder(nn.Module):
    """This class implements the Transformer decoder with embedding injections

    Arguments
    ---------
    num_layers : int
        Number of transformer layers for the decoder.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        Dimension of the model.
    kdim : int, optional
        Dimension for key (Optional).
    vdim : int, optional
        Dimension for value (Optional).
    dropout : float, optional
        Dropout for the decoder (Optional).
    emb_dropout: float, optional
        The speaker embedding dropout
    activation : Callable, optional
        The function to apply between layers, default nn.ReLU
    normalize_before : bool, optional
        Whether to normalize before layers.
    causal : bool
        Whether to allow future information in decoding.
    attention_type : str, optional
        Type of attention to use, "regularMHA" or "RelPosMHAXL"
    emb : dict
        Embedding configuration
    layerwise_renorm : bool, optional
        Whether to apply a normalization after each layer. If set to
        False, the normalization is applied at the end only

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = EmbeddingGuidedTransformerDecoder(
    ...     num_layers=1,
    ...     nhead=8,
    ...     d_ffn=1024,
    ...     d_model=512,
    ...     emb={
    ...         "spk": {
    ...             "kind": "pretrained",
    ...             "dim": 192,
    ...             "injection": "add"
    ...         }
    ...     }
    ... )
    >>> emb = torch.randn(8, 192)
    >>> output, _, _ = net(src, tgt, emb={"spk": emb})
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        emb_dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
        emb=None,
        layerwise_renorm=True,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        if emb is None:
            emb = {}
        self.emb_injection = _build_emb_injections(emb, d_model, num_layers)
        self.emb_norm = _build_emb_norms(emb)
        self.layerwise_renorm = layerwise_renorm
        self.emb_dropout = emb_dropout

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
        emb=None,
    ):
        """
        Arguments
        ----------
        tgt : torch.Tensor
            The sequence to the decoder layer (required).
        memory : torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask : torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt : torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src : torch.Tensor
            The positional embeddings for the source (optional).
        emb : dict
            An str->tensor dictionary of embeddings
        """
        output = tgt
        self_attns, multihead_attns = [], []

        if emb is not None:
            emb = _apply_emb_norm(emb, self.emb_norm, self.emb_dropout)

        for dec_layer, emb_injection in zip(self.layers, self.emb_injection):
            layer_input = output
            if emb is not None:
                for key, emb_item in emb.items():
                    output = emb_injection[key].before(output, emb_item)
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
            if emb is not None:
                for key, emb_item in emb.items():
                    output = emb_injection[key].after(
                        layer_input, output, emb_item
                    )
            # NOTE: This is a significant change after the last implementation - normalizing
            # at every layer
            if self.layerwise_renorm:
                output = self.norm(output)

        if not self.layerwise_renorm:
            output = self.norm(output)

        return output, self_attns, multihead_attns


def _build_emb_injection(emb_config, d_model):
    """Builds an embedding enjection module corresponding
    to the embedding configuration

    Arguments
    ---------
    emb_config : dict
        The embedding configuration

    d_model: int
        The model dimension

    Return
    ------
    injection : callable
        The injection module
    """
    emb_injection = emb_config.get("injection")
    emb_size = emb_config["dim"]
    if emb_injection is None:
        emb_injection = EmbeddingInjection.NULL
    if isinstance(emb_injection, str):
        emb_injection = EmbeddingInjection(emb_injection)
    if emb_injection in EMBEDDING_INJECTIONS:
        emb_injection = EMBEDDING_INJECTIONS[emb_injection](
            emb_size=emb_size,
            out_size=d_model,
        )
    return emb_injection


def _apply_emb_norm(emb, emb_norm, emb_dropout):
    if emb is not None:
        emb = {
            key: nn.functional.dropout(emb_norm[key](emb_item), emb_dropout)
            for key, emb_item in emb.items()
        }


def _build_emb_injections(emb, d_model, num_layers):
    return nn.ModuleList(
        [
            nn.ModuleDict(
                {
                    key: _build_emb_injection(emb_config, d_model)
                    for key, emb_config in emb.items()
                }
            )
            for _ in range(num_layers)
        ]
    )


def _build_emb_norms(emb):
    return nn.ModuleDict(
        {
            key: LayerNorm(
                input_size=emb_config["dim"],
            )
            for key, emb_config in emb.items()
        }
    )


class AdditiveEmbedding(nn.Module):
    """A simple embedding scheme where a projection of the embedding is computed
     and then added to the inputs. Outputs are passed straight through.

    Arguments
    ---------
    emb_size : int
        The embedding size
    out_size : int
        The output size
    """

    def __init__(self, emb_size, out_size):
        super().__init__()
        if emb_size == out_size:
            self.in_proj = nn.Identity()
        else:
            self.in_proj = Linear(
                input_size=emb_size,
                n_neurons=out_size,
            )
            nn.init.xavier_normal_(self.in_proj.w.weight)

    """A simple embedding mechanism that adds the embedding to the inputs before the layer"""

    def before(self, inputs, emb):
        """Injects embeddings into the model inputs

        Arguments
        ---------
        inputs : torch.Tensor
            The inputs
        emb : torch.Tensor
            The embedding tensor

        Returns
        -------
        out: torch.Tensor
            The output
        """
        emb_proj = self.in_proj(emb)
        return inputs + emb_proj.unsqueeze(1)

    def after(self, inputs, outputs, emb):
        """Injects embeddings into the model outputs

        Arguments
        ---------
        inputs : torch.Tensor
            The inputs
        outputs : torch.Tensor
            The outputs
        emb : torch.Tensor
            The embedding tensor

        Returns
        -------
        out: torch.Tensor
            The output
        """
        return outputs


class NullEmbedding(nn.Module):
    """A no-op embedding (causing the embedding to be
    completely ignored)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def before(self, inputs, emb):
        """Injects embeddings into the model inputs

        Arguments
        ---------
        inputs : torch.Tensor
            The inputs
        emb : torch.Tensor
            The embedding tensor

        Returns
        -------
        outputs : torch.Tensor
            The outputs, with embeddings injected
        """
        return inputs

    def after(self, inputs, outputs, emb):
        """Injects embeddings into the model outputs

        Arguments
        ---------
        inputs : torch.Tensor
            The inputs
        outputs : torch.Tensor
            The outputs
        emb : torch.Tensor
            The embedding tensor

        Returns
        -------
        outputs : torch.Tensor
            The outputs, with embeddings injected
        """
        return inputs


class EmbeddingInjection(Enum):
    """An enumeration for available injection mechanisms, to be used in
    configuration"""

    ADD = "add"
    NULL = "null"


EMBEDDING_INJECTIONS = {
    EmbeddingInjection.NULL: NullEmbedding,
    EmbeddingInjection.ADD: AdditiveEmbedding,
}


class DiscreteSSLVocoder(nn.Module):
    """A vocoder wrapper for Discrete SSL tokens

    Arguments
    ---------
    discrete_ssl : speechbrain.lobes.models.huggingface_transformers.discrete_ssl.DiscreteSSL
        a discrete SSL model
    layers : list, options
        the layers that are in use
    """

    def __init__(self, discrete_ssl, layers=None):
        super().__init__()
        self.discrete_ssl = discrete_ssl
        self.layers = layers

    def forward(self, tokens):
        """Takes a discretized input and returns its corresponding waveform

        Arguments
        ---------
        tokens : torch.Tensor
            Discrete SSL tokens

        Returns
        -------
        wav : torch.Tensor
            The waveform tensor"""
        wav = self.discrete_ssl.decode(tokens, SSL_layers=self.layers)
        return wav


def get_lookahead_mask_for_length(seq_len, device=None):
    """Creates a binary mask for each sequence which masks future frames.

    Arguments
    ---------
    seq_len: int
        The sequence length
    device : torch.Device | str
        The target device

    Returns
    -------
    mask : torch.Tensor
        Binary mask for masking future frames.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    mask = mask.detach()
    if device is not None:
        mask = mask.to(device)
    return mask


def get_silence_token(
    model,
    sample_length=100000,
    unsqueeze=False,
    device=None,
    model_kwargs=None,
):
    """Attempts to find out the silence tokens for a given model,
    if applicable

    Arguments
    ---------
    model : nn.Module
        A discrete token model, taking (wav, lengths) as arguments
    sample_length : int
        The length of the sample
    unsqueeze: bool
        Whether to add an extra dimension to the audio (needed for DAC)
    device : str | torch.Device
        The device to use
    model_kwargs : dict
        The arguments to pass to the model

    Returns
    -------
    silence_tokens : torch.Tensor
        The token(s) corresponding to silence

    silece_emb : torch.Tensor
        The embedding(s) corresponding to silence

    """
    if device is None:
        device = next(model.parameters()).device

    audio = torch.zeros(1, sample_length, device=device)
    if unsqueeze:
        audio = audio.unsqueeze(1)
    length = torch.ones(1, device=device)
    model_training = model.training
    model.eval()
    if model_kwargs is None:
        model_kwargs = {}
    tokens, *_ = model(audio, length, **model_kwargs)
    if model_training:
        model.train()
    tokens = tokens.squeeze(0)
    if unsqueeze:
        tokens = tokens.squeeze(0)
    silence_tokens = tokens.mode(0).values
    return silence_tokens


def get_silence_repr(model, sample_length=100000, layers=None, device=None):
    """Gets continuous silence

    Arguments
    ---------
    model : nn.Module
        A discrete token model, taking (wav, lengths) as arguments
    sample_length : int
        The length of the sample
    layers : list, optional
        The layers to return - if omitted, all layers will be returned
    device : str | torch.Device
        The device to use

    Returns
    -------
    silence : torch.Tensor
        A silecnce tensor
    """
    audio = torch.zeros(1, sample_length, device=device)
    length = torch.ones(1, device=device)
    audio_repr = model(audio, length)
    silence = audio_repr.squeeze(1).permute(1, 0, 2).mean(dim=0)
    if layers is not None:
        silence = silence[layers]
    return silence


def vocoder_to_device(vocoder, device):
    """Forces a vocoder to be moved to the specified device.
    Some vocoders do not respect the .to() command and need to be moved manually.

    Arguments
    ---------
    vocoder : torch.nn.Module
        the module

    device : str
        the device
    """
    if vocoder is not None:
        vocoder.to(device)
        if hasattr(vocoder, "device"):
            vocoder.device = device
        if hasattr(vocoder, "codec_vocoder"):
            vocoder.codec_vocoder.to(device)
            vocoder.codec_vocoder.device = device
