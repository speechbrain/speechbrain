"""A simplistic Text-to-Speech model operating on
discrete/tokenized audio representations, available in both
Transformer and RNN flavours.

NOTE: This model does not use the standard Transformer interface
in order to make it usable as both as a full model and as a
decoder-only model

Authors
* Artem Ploujnikov, 2023
"""

import torch
from torch import nn
from torch.nn import functional as F
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
    get_lookahead_mask,
)
from speechbrain.nnet.RNN import LSTM, GRU, AttentionalRNNDecoder
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.losses import kldiv_loss, distance_diff_loss
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
from speechbrain.nnet.embedding import MultiEmbedding
from speechbrain.dataio.dataio import length_to_mask
from collections import namedtuple
from tqdm.auto import tqdm

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
        "audio_tokens",
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
        "audio_tokens",
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
    audio_emb : torch.nn.Module, optional
        The audio embedding to be used
    activation : torch.nn.Module, optional
        The activation function to be used
    use_tgt_padding_mask : bool, optional
        whether to use a target padding mask
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
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
        target_dropout=None,
        audio_emb=None,
        audio_emb_size=128,
        activation=nn.LeakyReLU,
        use_tgt_padding_mask=False,
        audio_emb_freeze=False,
        max_decoder_steps=1000,
        bos_idx=0,
        gate_threshold=0.5,
        gate_offset=0,
        show_inference_progress=True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens_per_step = tokens_per_step
        self.dec = TransformerDecoder(
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
        self.tgt_in_proj = Linear(
            input_size=audio_emb_size * tokens_per_step, n_neurons=d_model,
        )
        self.out_proj = Linear(
            input_size=d_model, n_neurons=num_tokens * tokens_per_step,
        )
        self.gate = Linear(input_size=d_model, n_neurons=1)
        if audio_emb is None:
            audio_emb = MultiEmbedding(
                num_embeddings=num_tokens,
                embedding_dim=audio_emb_size,
                num_heads=tokens_per_step,
                normalized=True,
                d_model=d_model,
            )
        self.positional_encoding = PositionalEncoding(
            d_model, max_decoder_steps
        )
        if target_dropout is None:
            target_dropout = dropout
        self.target_dropout = target_dropout
        self.audio_emb = audio_emb
        self.max_decoder_steps = max_decoder_steps
        self.attention_type = attention_type
        self.use_tgt_padding_mask = use_tgt_padding_mask
        self.audio_emb_freeze = audio_emb_freeze
        self.bos_idx = bos_idx
        self.gate_threshold = gate_threshold
        self.gate_offset = gate_offset
        self.show_inference_progress = show_inference_progress
        if self.audio_emb_freeze:
            for parameter in self.audio_emb.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        enc_out,
        tgt,
        src_length=None,
        src_key_padding_mask=None,
        tgt_length=None,
        tgt_key_padding_mask=None,
        pos_embs_src=None,
        context=None,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        src : torch.Tensor
            Raw encoder outputs
        tgt : torch.Tensor
            Targets (audio tokens)
        src_length : torch.Tensor
            The relative lengths of the source sequence
        tgt_length : torch.Tensor
            Target lengths
        pos_embs_src : dict
            Source positional embeddings
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

        batch_size, audio_max_len, heads, audio_dim = audio_emb.shape
        audio_emb_combined = audio_emb.reshape(
            batch_size, audio_max_len, heads * audio_dim
        )
        tgt = self.tgt_in_proj(audio_emb_combined)
        tgt = F.dropout(tgt, self.target_dropout, training=self.training)

        tgt_mask = get_lookahead_mask(tgt)
        if self.attention_type == "RelPosMHAXL":
            pos_embs_tgt = self.positional_encoding(tgt)
        else:
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_tgt = None
        (dec_out, dec_self_attn, dec_attn,) = self.dec(
            tgt=tgt,
            memory=enc_out,
            memory_mask=None,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_tgt,
            pos_embs_src=pos_embs_src,
        )

        lin_out = self.out_proj(dec_out)
        batch_size, audio_max_len, _ = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size, audio_max_len, self.tokens_per_step, self.num_tokens,
        )
        gate_out = self.gate(dec_out).squeeze(-1)
        return TokotronDecoderOutput(
            lin_out_heads,
            gate_out,
            dec_self_attn,
            dec_attn,
            get_alignments(dec_attn),
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

    def infer(self, enc_out, length):
        """Performs autoregressive inference

        Arguments
        ---------
        decoder : callable
            The decoder module

        enc_out : torch.Tensor
            Raw encoder outputs

        length : torch.Tensor
            Relative lengths

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
        """
        with torch.no_grad():
            context = {}
            batch_size = enc_out.size(0)

            # Initialize BOS
            bos = get_bos(
                batch_size,
                self.tokens_per_step,
                self.bos_idx,
                device=enc_out.device,
            )
            audio_tokens = bos
            audio_tokens_length = torch.ones(batch_size, device=enc_out.device)
            steps_range = range(self.max_decoder_steps)

            # Initialize the gate activation index
            seq_gate_idx = (
                torch.ones(batch_size, device=enc_out.device)
                * self.max_decoder_steps
            )

            # Initialize an indicator that tells whether the gate has activated
            # for a given sample
            seq_gate_act = torch.zeros(batch_size, device=enc_out.device).bool()

            # Show progress if enabled
            if self.show_inference_progress:
                steps_range = tqdm(steps_range, desc="Inference")
            for idx in steps_range:
                # One autoregressive step
                step_out = self.forward(
                    enc_out=enc_out,
                    src_length=length,
                    tgt=audio_tokens,
                    tgt_length=audio_tokens_length,
                    context=context,
                )
                audio_tokens_out = step_out.out.argmax(-1)

                # The model outputs predictions without BOS. Add the BOS back for the
                # following step
                audio_tokens = torch.cat([bos, audio_tokens_out], dim=1)
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
                seq_done = seq_gate_act & (
                    seq_gate_idx - idx >= self.gate_offset
                )

                # Terminate inference if all samples are done
                done = seq_done.all()
                if done.item():
                    break

            # Length = gate activation index + the offset, not exceeding
            length_abs = (seq_gate_idx + self.gate_offset).clip(
                max=self.max_decoder_steps
            )
            # Compute relative lengths
            length = length_abs.float() / audio_tokens_out.size(1)

        return TokotronDecoderInfernceOutput(
            audio_tokens=audio_tokens_out,
            length=length,
            dec_self_attn=step_out.dec_self_attn,
            dec_attn=step_out.dec_attn,
            alignments=step_out.alignments,
            p_eos=step_out.gate_out.sigmoid(),
        )


class TokotronTransformerModel(nn.Module):
    """An end-to-end Tokotron model receiving characters or phonemes
    as inputs and outputting audio tokens

    Arguments
    ---------
    input_num_tokens : int
        The number of input characters or phonemes available
    audio_num_tokens : int
        The number of audio tokens
    audio_tokens_per_step : int
        The number of output audio tokens per tranformer step.
        When using Vocodec, this corresponds to the number of
        quantizers in the model used
    d_model : int
        The number of expected features in the encoder/decoder inputs (default=512).
    d_ffn : int, optional
        The dimension of the feedforward network model hidden layer.
    nhead : int
        The number of heads in the multi-head attention models (default=8).
    enc_num_layers : int, optional
        The number of encoder layers in1ì the encoder.
    dec_num_layers : int, optional
        The number of decoder layers in the decoder.
    dropout : int, optional
        The dropout value.
    target_dropout : float, optional
        The dropout probability for targets
    activation : torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    bos_idx : int
        the Beginning-of-Sequence index
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    max_audio_length: int
        The maximum number of tokens to be output
    use_tgt_padding_mask : bool, optional
        Whether to use a target padding mask
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
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
        activation=nn.LeakyReLU,
        max_audio_length=1000,
        bos_idx=0,
        gate_threshold=0.5,
        gate_offset=0,
        use_tgt_padding_mask=False,
        audio_emb_freeze=False,
        show_inference_progress=True,
        vocoder=None,
    ):
        super().__init__()
        self.in_emb = Embedding(
            num_embeddings=input_num_tokens, embedding_dim=d_model,
        )
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
        self.decoder = TokotronTransformerDecoder(
            num_tokens=audio_num_tokens,
            tokens_per_step=audio_tokens_per_step,
            d_model=d_model,
            d_ffn=d_ffn,
            nhead=nhead,
            attention_type=attention_type,
            num_layers=dec_num_layers,
            activation=activation,
            dropout=dropout,
            target_dropout=target_dropout,
            use_tgt_padding_mask=use_tgt_padding_mask,
            audio_emb_freeze=audio_emb_freeze,
            max_decoder_steps=max_audio_length,
            bos_idx=bos_idx,
            gate_threshold=gate_threshold,
            gate_offset=gate_offset,
            show_inference_progress=show_inference_progress,
        )
        self.bos_idx = bos_idx
        self.vocoder = vocoder
        self.attention_type = attention_type
        self.gate_offset = gate_offset
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_audio_length
            )

    @property
    def gate_offset(self):
        return self.decoder.gate_offset

    @gate_offset.setter
    def gate_offset(self, value):
        self.decoder.gate_offset = value

    def forward(
        self, input_tokens, input_length, audio_tokens, audio_length,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        audio_tokens : torch.Tensor
            a (Batch x Length) tensor of output audio tokens (e.g. encodec)
        audio_length : torch.Tensor
            a 1-D tensor of relative output lengths"""

        src, src_key_padding_mask, pos_embs_encoder = self.process_inputs(
            input_tokens, input_length
        )

        enc_out, enc_self_attn = self.encoder(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        dec_out = self.decoder(
            enc_out=enc_out,
            tgt=audio_tokens,
            tgt_length=audio_length,
            src_length=input_length,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs_src=pos_embs_encoder,
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
            input_length * input_max_len, input_max_len,
        ).logical_not()
        return src, src_key_padding_mask, pos_embs_encoder

    def infer(self, input_tokens, input_length):
        """Performs end-to-end inference

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

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
        dec_out = self.decoder.infer(enc_out, input_length)
        wav, wav_length = None, None
        if self.vocoder is not None:
            wav, wav_length = self.vocoder(dec_out.audio_tokens, input_length)
        return TokotronInfernceOutput(
            audio_tokens=dec_out.audio_tokens,
            length=dec_out.audio_tokens,
            wav=wav,
            wav_length=wav_length,
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


class TokotronRNNModel(nn.Module):
    """
    An attention sequence-to-sequence RNN-based Tokotron implementation

    Arguments
    ---------
    input_num_tokens : int
        The number of input characters or phonemes available
    audio_num_tokens : int
        The number of audio tokens
    audio_tokens_per_step : int
        The number of output audio tokens per tranformer step.
        When using Vocodec, this corresponds to the number of
        quantizers in the model used
    bos_idx : int
        the Beginning-of-Sequence index
    gate_threshold : int
        The minimum gate value (post-sigmoid) to consider the sequence
        as complete during auto-regressive inference
    gate_offset : int, optional
        The number of steps from the gate activation threshold until inference
        stops. By default, inference stops immediately. This parameter is useful
        for "soft" gate implementations where the gate starts outputting positive
        probabilities before actual EOS
    max_audio_length: int
        The maximum number of tokens to be output
    """

    def __init__(
        self,
        input_num_tokens,
        input_emb_size=512,
        audio_num_tokens=1024,
        audio_tokens_per_step=2,
        enc_rnn_type="gru",
        enc_hidden_size=512,
        enc_num_layers=6,
        enc_bidirectional=True,
        dec_rnn_type="gru",
        dec_input_size=512,
        dec_hidden_size=512,
        dec_num_layers=6,
        dec_attn_type="location",
        dec_attn_dim=512,
        dec_attn_kernel_size=100,
        dec_attn_channels=10,
        dec_attn_scaling=1.0,
        dropout=0.2,
        bos_idx=0,
        max_audio_length=1000,
        gate_threshold=0.5,
        gate_offset=0.0,
        audio_emb_freeze=False,
        show_inference_progress=True,
        vocoder=None,
    ):
        super().__init__()
        self.input_num_tokens = input_num_tokens
        self.audio_num_tokens = audio_num_tokens
        self.audio_tokens_per_step = audio_tokens_per_step
        self.in_emb = Embedding(
            num_embeddings=input_num_tokens, embedding_dim=input_emb_size,
        )
        enc_module = _rnn_modules.get(enc_rnn_type, enc_rnn_type)
        if not issubclass(enc_module, nn.Module):
            supported_modules = ",".join(_rnn_modules.keys())
            raise ValueError(
                f"Supported values for enc_rnn_type: {supported_modules} or "
                "an instance of nn.Module"
            )
        self.encoder = enc_module(
            input_size=input_emb_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_num_layers,
            bidirectional=enc_bidirectional,
            dropout=dropout,
        )
        enc_dim = enc_hidden_size
        if enc_bidirectional:
            enc_dim *= 2
        self.decoder = TokotronRNNDecoder(
            num_tokens=audio_num_tokens,
            tokens_per_step=audio_tokens_per_step,
            rnn_type=dec_rnn_type,
            enc_dim=enc_dim,
            input_size=dec_input_size,
            hidden_size=dec_hidden_size,
            num_layers=dec_num_layers,
            attn_type=dec_attn_type,
            attn_dim=dec_attn_dim,
            attn_kernel_size=dec_attn_kernel_size,
            attn_channels=dec_attn_channels,
            attn_scaling=dec_attn_scaling,
            audio_emb_freeze=audio_emb_freeze,
            max_decoder_steps=max_audio_length,
            bos_idx=bos_idx,
            gate_threshold=gate_threshold,
            gate_offset=gate_offset,
            show_inference_progress=show_inference_progress,
        )

        self.vocoder = vocoder

    def forward(
        self, input_tokens, input_length, audio_tokens, audio_length,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths
        audio_tokens : torch.Tensor
            a (Batch x Length) tensor of output audio tokens (e.g. encodec)
        audio_length : torch.Tensor
            a 1-D tensor of relative output lengths"""

        src = self.in_emb(input_tokens)
        enc_out, _ = self.encoder(src)
        dec_out = self.decoder(
            enc_out=enc_out,
            tgt=audio_tokens,
            tgt_length=audio_length,
            src_length=input_length,
        )
        return TokotronOutput(
            out=dec_out.out,
            gate_out=dec_out.gate_out,
            p_eos=dec_out.gate_out.sigmoid(),
            enc_self_attn=None,
            dec_self_attn=None,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
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

    def infer(self, input_tokens, input_length):
        """Performs end-to-end inference

        Arguments
        ---------
        input_tokens : torch.Tensor
            a (Batch x Length) tensor of input tokens, representing
            characters or phonemes
        input_length : torch.Tensor
            a 1-D tensor of relative input lengths

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

        """
        src = self.in_emb(input_tokens)
        enc_out, _ = self.encoder(src)
        dec_out = self.decoder.infer(enc_out, input_length)
        wav, wav_length = None, None
        if self.vocoder is not None:
            wav, wav_length = self.vocoder(dec_out.audio_tokens, input_length)
        return TokotronInfernceOutput(
            audio_tokens=dec_out.audio_tokens,
            length=dec_out.audio_tokens,
            wav=wav,
            wav_length=wav_length,
            enc_self_attn=None,
            dec_self_attn=None,
            dec_attn=dec_out.dec_attn,
            alignments=dec_out.alignments,
            p_eos=dec_out.p_eos,
        )


class TokotronRNNDecoder(nn.Module):
    """The Tokotron decoder - can be used in a standalone model or as
    a component of a larger model

    Arguments
    ---------
    num_tokens : int, optional
        The number of tokens
    tokens_per_step : int, optional
        The number of tokens to be output, per transformer time step
    audio_emb : nn.Module, optional
        The audio embedding module
    audio_emb_size : int
        The audio emedding size
    rnn_type : str|nn.Module
        The type of RNN to be used
    hidden_size : int
        The number of neurons in RNN hidden layers
    attn_type : str
        The type of attention to use (location, content).
    attn_dim : int
        Number of attention module internal and output neurons.
    attn_kernel_size : int
        The kernel size for location-aware attention
    attn_channels : int
        The number of channels for location-aware attention
    attn_scaling : float
        The scaling factor for location-aware attention
    audio_emb_freeze : bool, optional
        Whether audio embeddings should be frozen
    """

    def __init__(
        self,
        num_tokens=1024,
        tokens_per_step=2,
        rnn_type="gru",
        enc_dim=1024,
        input_size=512,
        hidden_size=512,
        num_layers=6,
        attn_type="location",
        attn_dim=512,
        attn_kernel_size=100,
        attn_channels=10,
        attn_scaling=1.0,
        nonlinearity="relu",
        normalization="batchnorm",
        dropout=0.2,
        target_dropout=None,
        audio_emb=None,
        audio_emb_size=128,
        audio_emb_freeze=False,
        max_decoder_steps=1000,
        bos_idx=0,
        gate_threshold=0.5,
        gate_offset=0,
        show_inference_progress=True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens_per_step = tokens_per_step
        if audio_emb is None:
            audio_emb = MultiEmbedding(
                num_embeddings=num_tokens,
                embedding_dim=audio_emb_size,
                num_heads=tokens_per_step,
                normalized=False,
            )
        self.audio_emb = audio_emb
        if target_dropout is None:
            target_dropout = dropout
        self.target_dropout = target_dropout
        self.dec = AttentionalRNNDecoder(
            rnn_type=rnn_type,
            input_size=input_size,
            enc_dim=enc_dim,
            attn_type=attn_type,
            attn_dim=attn_dim,
            kernel_size=attn_kernel_size,
            channels=attn_channels,
            scaling=attn_scaling,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            nonlinearity=nonlinearity,
            normalization=normalization,
        )
        self.tgt_in_proj = Linear(
            input_size=audio_emb_size * tokens_per_step, n_neurons=input_size,
        )
        self.out_proj = Linear(
            input_size=hidden_size, n_neurons=num_tokens * tokens_per_step,
        )
        self.gate = Linear(input_size=hidden_size, n_neurons=1)
        self.audio_emb_freeze = audio_emb_freeze
        self.max_decoder_steps = max_decoder_steps
        self.bos_idx = bos_idx
        self.gate_threshold = gate_threshold
        self.gate_offset = gate_offset
        self.show_inference_progress = show_inference_progress
        if self.audio_emb_freeze:
            for parameter in self.audio_emb.parameters():
                parameter.requires_grad_(False)

    @property
    def gate_offset(self):
        return self.decoder.gate_offset

    @gate_offset.setter
    def gate_offset(self, value):
        self.decoder.gate_offset = value

    def forward(
        self, enc_out, tgt, src_length=None, tgt_length=None,
    ):
        """Computes the forward pass, for training

        Arguments
        ---------
        src : torch.Tensor
            Raw encoder outputs
        tgt : torch.Tensor
            Targets (audio tokens)
        src_length : torch.Tensor
            The relative lengths of the source sequence
        tgt_length : torch.Tensor
            Target lengths
        """
        audio_emb = self.audio_emb(tgt)

        batch_size, audio_max_len, heads, audio_dim = audio_emb.shape
        audio_emb_combined = audio_emb.reshape(
            batch_size, audio_max_len, heads * audio_dim
        )
        tgt = self.tgt_in_proj(audio_emb_combined)
        tgt = F.dropout(tgt, self.target_dropout, training=self.training)

        dec_out, dec_attn = self.dec(tgt, enc_out, src_length)
        lin_out = self.out_proj(dec_out)
        batch_size, audio_max_len, _ = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size, audio_max_len, self.tokens_per_step, self.num_tokens,
        )
        gate_out = self.gate(dec_out).squeeze(-1)
        return TokotronDecoderOutput(
            lin_out_heads,
            gate_out,
            None,
            dec_attn,
            get_alignments(dec_attn),
            {},
        )

    def forward_step(self, enc_out, tgt, src_length=None, context=None):
        """
        Performs a single RNN step - used in inference

        src : torch.Tensor
            Raw encoder outputs
        tgt : torch.Tensor
            Targets (audio tokens)
        src_length : torch.Tensor
            The relative lengths of the source sequence
        """
        audio_emb = self.audio_emb(tgt)

        batch_size, heads, audio_dim = audio_emb.shape
        audio_emb_combined = audio_emb.reshape(batch_size, heads * audio_dim)
        tgt = self.tgt_in_proj(audio_emb_combined)
        if context is None:
            context = {
                "c": torch.zeros(
                    enc_out.shape[0], self.dec.attn_dim, device=enc_out.device
                ),
                "hs": None,
            }
        dec_out, hs, c, w = self.dec.forward_step(
            tgt, context["hs"], context["c"], enc_out, src_length
        )
        context["hs"] = hs
        context["c"] = c
        dec_attn = w
        lin_out = self.out_proj(dec_out)
        batch_size, _ = lin_out.shape
        lin_out_heads = lin_out.reshape(
            batch_size, self.tokens_per_step, self.num_tokens,
        )

        gate_out = self.gate(dec_out).squeeze(-1)

        return TokotronDecoderOutput(
            lin_out_heads, gate_out, None, dec_attn, dec_attn, context
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

    def infer(self, enc_out, length):
        """Performs autoregressive inference

        Arguments
        ---------
        decoder : callable
            The decoder module

        enc_out : torch.Tensor
            Raw encoder outputs

        length : torch.Tensor
            Relative lengths

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
        """
        with torch.no_grad():
            context = None
            batch_size = enc_out.size(0)

            # Initialize BOS
            bos = get_bos(
                batch_size,
                self.tokens_per_step,
                self.bos_idx,
                device=enc_out.device,
            ).squeeze(1)

            steps_range = range(self.max_decoder_steps)

            # Initialize the gate activation index
            seq_gate_idx = (
                torch.ones(batch_size, device=enc_out.device)
                * self.max_decoder_steps
            )

            # Initialize an indicator that tells whether the gate has activated
            # for a given sample
            seq_gate_act = torch.zeros(batch_size, device=enc_out.device).bool()

            step_tokens = bos
            # Show progress if enabled
            if self.show_inference_progress:
                steps_range = tqdm(steps_range, desc="Inference")

            audio_tokens_lst = []
            dec_attn_lst = []
            p_eos_lst = []

            for idx in steps_range:
                # One autoregressive step
                step_out = self.forward_step(
                    enc_out=enc_out,
                    src_length=length,
                    tgt=step_tokens,
                    context=context,
                )

                step_tokens = step_out.out.argmax(-1)

                # Compute the gate activation (final sigmoid)
                step_gate_act = (
                    step_out.gate_out.sigmoid() > self.gate_threshold
                )

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
                        torch.tensor(idx, device=step_gate_act.device),
                        seq_gate_idx,
                    ),
                )

                audio_tokens_lst.append(step_tokens)
                dec_attn_lst.append(step_out.dec_attn)
                p_eos_lst.append(step_gate_act)

                # Update the gate indicator
                seq_gate_act = seq_gate_act | step_gate_act

                # For a given sample, consider it done if the gate has activated at least
                # gate_offset steps ago
                seq_done = seq_gate_act & (
                    seq_gate_idx - idx >= self.gate_offset
                )

                # Terminate inference if all samples are done
                done = seq_done.all()
                if done.item():
                    break

            # Concatenate outputs for all steps
            audio_tokens_out = torch.stack(audio_tokens_lst, dim=1)
            dec_attn = torch.stack(dec_attn_lst, dim=1)
            p_eos = torch.stack(p_eos_lst, dim=1)

            # Length = gate activation index + the offset, not exceeding
            length_abs = (seq_gate_idx + self.gate_offset).clip(
                max=self.max_decoder_steps
            )
            # Compute relative lengths
            length = length_abs.float() / audio_tokens_out.size(1)

        return TokotronDecoderInfernceOutput(
            audio_tokens=audio_tokens_out,
            length=length,
            dec_self_attn=None,
            dec_attn=dec_attn,
            alignments=dec_attn,
            p_eos=p_eos,
        )


_rnn_modules = {"gru": GRU, "lstm": LSTM}


def get_bos(batch_size, tokens_per_step, bos_idx, device="cpu"):
    """Constructs a beginning-of-sequence (BOS) sequence for
    autoregressive inference

    Arguments
    ---------
    batch_size : int
        The size of the batch dimension
    device : str|torch.Device
        The device identifier

    Returns
    -------
    seq: torch.Tensor
        the target sequence"""
    return torch.ones(batch_size, 1, tokens_per_step, device=device) * bos_idx


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
    tagrets : torch.Tensor
        Targets for gate outputs - EOS positions are marked as 1,
        non-EOS positions are marked at 0
    weights : torch.Tensor
        Weights by which individual position losses will be multiplied
    """
    pos = torch.arange(out_len, device=lengths.device)[None, :]
    gate_targets = pos >= (lengths * out_len)[:, None]
    gate_weights = torch.where(
        gate_targets, 0.5 / (1.0 - lengths)[:, None], 0.5 / lengths[:, None],
    )
    return gate_targets.float(), gate_weights


def get_alignments(attn):
    """Aggregates alignments from multiple layers and heads

    Arguments
    ---------
    attn: list
        raw attentions returned from a Transformer

    Results
    -------
    alignments: torch.Tensor
        The resulting alignments
    """
    return torch.cat([item.unsqueeze(-1) for item in attn], dim=-1).mean(dim=-1)


TokotronLossDetails = namedtuple(
    "TokotronLossDetails", ["loss", "seq_loss", "gate_loss", "attn_loss"]
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
    ):
        super().__init__()
        self.guided_attention_weight = guided_attention_weight
        self.gate_weight = gate_weight
        self.gate_beta = gate_beta
        self.gate_gamma = gate_gamma
        self.gate_max_weight = gate_max_weight
        self.silence_padding = silence_padding
        if seq_cost is None:
            seq_cost = kldiv_loss
        self.seq_cost = seq_cost
        self.attn_cost = GuidedAttentionLoss(sigma=guided_attention_sigma,)

    def forward(
        self,
        predictions,
        audio_tokens,
        audio_length,
        input_tokens,
        input_length,
        reduction="mean",
    ):
        p_seq = predictions.out.log_softmax(dim=-1)
        batch_size, out_len, heads, tok_dim = p_seq.shape
        max_len = out_len - 1
        p_seq_reshaped = (
            p_seq.transpose(1, 2).reshape(batch_size * heads, out_len, tok_dim)
        )[:, :max_len, :]
        audio_tokens_reshaped = audio_tokens.transpose(1, 2).reshape(
            batch_size * heads, max_len
        )
        lengths_reshaped = audio_length.repeat(heads)
        seq_loss = self.seq_cost(
            p_seq_reshaped,
            audio_tokens_reshaped,
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
        loss = (
            seq_loss
            + self.guided_attention_weight * attn_loss
            + self.gate_weight * gate_loss
        )
        return TokotronLossDetails(loss, seq_loss, gate_loss, attn_loss)
