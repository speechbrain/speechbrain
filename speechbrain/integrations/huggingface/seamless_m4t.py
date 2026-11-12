"""This lobe enables the integration of HuggingFace pretrained SeamlessM4T models
for speech-to-text tasks (speech recognition and speech translation).

Reference: https://arxiv.org/abs/2308.11596 (SeamlessM4T)
Reference: https://arxiv.org/abs/2312.05187 (SeamlessM4T v2)
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html
The SeamlessM4T tokenizer also needs protobuf (``pip install protobuf``).
The key-value cache of ``S2SSeamlessBeamSearcher`` needs transformers >= 4.54
(older versions return the cache as tuples); with older versions, set
``use_kv_cache=False`` in the beam searcher.

Authors
 * Othman Istaiteh 2026
"""

import torch

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.integrations.huggingface.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

# Groups of sub-modules (prefixes of ``model.named_modules()``) that can be
# frozen with a single flag. The input embeddings are tied with the lm_head.
FREEZABLE_GROUPS = {
    "encoder": ["speech_encoder"],
    "feature_projection": ["speech_encoder.feature_projection"],
    "adapter": [
        "speech_encoder.intermediate_ffn",
        "speech_encoder.adapter",
        "speech_encoder.inner_layer_norm",
    ],
    "decoder": ["text_decoder", "lm_head", "shared"],
    "embeddings": ["shared", "text_decoder.embed_tokens", "lm_head"],
}


class SeamlessM4T(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained SeamlessM4T
    (v1 and v2) models for speech-to-text tasks.

    Only the speech encoder and the text decoder are loaded (the
    ``SeamlessM4T[v2]ForSpeechToText`` sub-model of transformers), so the text
    encoder and the speech generation modules never occupy memory.

    The same model performs speech recognition and speech translation: the
    task is selected by the target language. For ASR, set ``target_lang`` to
    the language spoken in the audio; for ST, set it to the language of the
    translation. Language codes follow the Seamless convention (3 letters,
    e.g. "eng", "fra", "arb").

    The decoder is fed with ``[decoder_start, target_lang, tokens...]``, as in
    the original model. To stay compatible with the SpeechBrain searchers,
    ``decoder_start`` is prepended internally: recipes and searchers only deal
    with ``[target_lang, tokens...]`` (see ``bos`` and ``encode_target``).
    Decoding uses ``speechbrain.decoders.seq2seq.S2SSeamlessGreedySearcher``
    and ``S2SSeamlessBeamSearcher``, with key-value caching.

    LayerDrop is disabled (see ``override_config``), as layers skipped at
    random leave parameters without gradients, which DDP does not allow.

    With v1 checkpoints, the convolution modules of the speech encoder use
    BatchNorm: its running statistics are still updated when the modules are
    in training mode, even if the encoder is frozen (v2 uses LayerNorm).

    Arguments
    ---------
    source : str
        HuggingFace hub name or local path, e.g. "facebook/seamless-m4t-v2-large".
    save_path : str
        Path (dir) of the downloaded model.
    target_lang : str (default: "eng")
        Seamless code of the language the decoder must produce.
    sampling_rate : int (default: 16000)
        Sampling rate of the input audio.
    encoder_only : bool (default: False)
        If True, the text decoder is deleted and the forward function returns
        the speech encoder outputs only.
    freeze : bool (default: False)
        If True, the whole model is frozen and run without gradients.
    freeze_encoder : bool (default: False)
        If True, the speech encoder (including its adapter) is frozen.
    freeze_feature_projection : bool (default: False)
        If True, the feature projection at the input of the speech encoder is
        frozen.
    freeze_adapter : bool (default: False)
        If True, the length adapter between the encoder and the decoder is
        frozen.
    freeze_decoder : bool (default: False)
        If True, the text decoder (including the embeddings and lm_head) is
        frozen.
    freeze_embeddings : bool (default: False)
        If True, the token embeddings and the tied lm_head are frozen. They
        hold a large part of the parameters (256k entries), so freezing them
        saves a lot of memory with little impact on fine-tuning.
    frozen_modules : list of str (default: None)
        Extra sub-modules to freeze, given as prefixes of the HuggingFace
        module names, e.g. ["speech_encoder.encoder.layers.0"].
    output_attentions : bool (default: False)
        If True, the decoder returns the last cross-attention weights. The
        model is then loaded with the eager attention, as the fused attention
        kernels do not return the weights, so it is slower.
    gradient_checkpointing : bool (default: False)
        If True, enables (non-reentrant) gradient checkpointing to reduce the
        activation memory during fine-tuning.
    **kwargs
        Passed to ``HFTransformersInterface`` (e.g. ``cache_dir``), and from
        there to the ``from_pretrained`` method of transformers.

    Example
    -------
    >>> model_hub = "facebook/hf-seamless-m4t-medium"
    >>> save_path = "savedir"
    >>> model = SeamlessM4T(model_hub, save_path, target_lang="fra")
    >>> wav = torch.randn([2, 16000])
    >>> wav_lens = torch.tensor([1.0, 0.75])
    >>> tokens_bos = torch.tensor([[model.bos]] * 2)
    >>> enc_out, enc_lens, logits = model(wav, tokens_bos, wav_lens)
    >>> logits.shape[:2]
    torch.Size([2, 1])
    """

    def __init__(
        self,
        source,
        save_path,
        target_lang="eng",
        sampling_rate=16000,
        encoder_only=False,
        freeze=False,
        freeze_encoder=False,
        freeze_feature_projection=False,
        freeze_adapter=False,
        freeze_decoder=False,
        freeze_embeddings=False,
        frozen_modules=None,
        output_attentions=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        if output_attentions:
            kwargs["attn_implementation"] = "eager"
        super().__init__(
            source=source, save_path=save_path, freeze=freeze, **kwargs
        )
        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.output_attentions = output_attentions

        self.load_feature_extractor(
            source, save_path, sampling_rate=sampling_rate
        )

        if encoder_only:
            self.tokenizer = None
            self.model.text_decoder = None
            self.model.lm_head = None
            self.model.shared = None
        else:
            self.load_tokenizer(source, cache_dir=save_path)
            self.set_target_lang(target_lang)

        if not self.freeze:
            flags = {
                "encoder": freeze_encoder,
                "feature_projection": freeze_feature_projection,
                "adapter": freeze_adapter,
                "decoder": freeze_decoder and not encoder_only,
                "embeddings": freeze_embeddings and not encoder_only,
            }
            prefixes = [
                prefix
                for group, enabled in flags.items()
                if enabled
                for prefix in FREEZABLE_GROUPS[group]
            ]
            self._freeze_submodules(prefixes + list(frozen_modules or []))

        if gradient_checkpointing and not self.freeze:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def override_config(self, config):
        """Disables LayerDrop, so that every trainable parameter gets a
        gradient at each step, as DDP requires.

        Arguments
        ---------
        config : SeamlessM4TConfig or SeamlessM4Tv2Config
            The original config.

        Returns
        -------
        config : SeamlessM4TConfig or SeamlessM4Tv2Config
            The config without LayerDrop.
        """
        config.speech_encoder_layerdrop = 0.0
        config.encoder_layerdrop = 0.0
        config.decoder_layerdrop = 0.0
        return config

    def _from_pretrained(self, source, save_path, cache_dir, **kwargs):
        """Loads the speech-to-text sub-model matching the checkpoint version.

        Arguments
        ---------
        source : str
            HuggingFace hub name or local path.
        save_path : str
            Path (dir) of the downloaded model.
        cache_dir : str
            Path (dir) in which a downloaded pretrained model configuration
            should be cached.
        **kwargs
            Extra keyword arguments passed to the `from_pretrained` function.
        """
        from transformers import (
            SeamlessM4TForSpeechToText,
            SeamlessM4Tv2ForSpeechToText,
        )

        speech_to_text_classes = {
            "seamless_m4t": SeamlessM4TForSpeechToText,
            "seamless_m4t_v2": SeamlessM4Tv2ForSpeechToText,
        }
        if self.config.model_type not in speech_to_text_classes:
            raise ValueError(
                f"{source} is a {self.config.model_type} model, "
                f"expected one of {list(speech_to_text_classes)}."
            )
        self.auto_class = speech_to_text_classes[self.config.model_type]
        super()._from_pretrained(source, save_path, cache_dir, **kwargs)

    def _freeze_submodules(self, prefixes):
        """Freezes the sub-modules matching the prefixes with ``freeze_model``.

        Arguments
        ---------
        prefixes : list of str
            Names of HuggingFace sub-modules (``model.named_modules()``).

        Raises
        ------
        ValueError
            If a prefix does not match any sub-module.
        """
        modules = dict(self.model.named_modules())
        for prefix in dict.fromkeys(prefixes):
            if prefix not in modules:
                raise ValueError(
                    f"Cannot freeze '{prefix}': no such sub-module in "
                    f"{type(self.model).__name__}."
                )
            logger.warning(
                f"speechbrain.integrations.huggingface.seamless_m4t - {prefix} is frozen."
            )
            self.freeze_model(modules[prefix])

    def set_target_lang(self, target_lang):
        """Sets the language produced by the decoder.

        Arguments
        ---------
        target_lang : str
            Seamless language code, e.g. "fra".

        Raises
        ------
        KeyError
            If the language is not in the tokenizer vocabulary.
        """
        token = f"__{target_lang}__"
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise KeyError(f"Language {target_lang} not found in tokenizer.")
        self.target_lang = target_lang
        self.tokenizer.tgt_lang = target_lang

    @property
    def bos(self):
        """Token id of the target language, used as first decoder input."""
        return self.tokenizer.convert_tokens_to_ids(f"__{self.target_lang}__")

    @property
    def eos(self):
        """Token id of the end of sequence."""
        return self.config.eos_token_id

    @property
    def pad(self):
        """Token id of the padding."""
        return self.config.pad_token_id

    @property
    def decoder_start(self):
        """Token id prepended internally to every decoder input."""
        return self.config.decoder_start_token_id

    @property
    def vocab_size(self):
        """Size of the output vocabulary."""
        return self.config.vocab_size

    def encode_target(self, text):
        """Tokenizes a target sentence as ``[target_lang] + tokens + [eos]``.

        ``[:-1]`` gives the decoder input and ``[1:]`` gives the targets.

        Arguments
        ---------
        text : str
            The transcription or translation.

        Returns
        -------
        list of int
            The token ids.
        """
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        return [self.bos] + tokens + [self.eos]

    def forward(self, wav, decoder_input_ids=None, wav_lens=None):
        """Computes the speech features, encodes them and runs the decoder.

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals, shape (batch, time).
        decoder_input_ids : torch.Tensor
            Decoder inputs starting with the target language token,
            shape (batch, tokens). Must be None if ``encoder_only``.
        wav_lens : torch.Tensor
            Relative lengths of the signals in SpeechBrain format.

        Returns
        -------
        enc_out : torch.Tensor
            The speech encoder outputs, shape (batch, frames, hidden).
        enc_lens : torch.Tensor
            Relative lengths of the encoder outputs. They are the lengths to
            give to the SpeechBrain searchers.
        logits : torch.Tensor
            The decoder logits, shape (batch, tokens, vocab). Only returned if
            the model is not ``encoder_only``.
        """
        if self.encoder_only and decoder_input_ids is not None:
            raise ValueError(
                "The encoder_only model takes no decoder_input_ids; "
                "give the lengths as wav_lens=..."
            )

        def _forward():
            """Forward pass of the model"""
            input_features, attention_mask = self.compute_features(
                wav, wav_lens
            )
            enc_out, enc_lens = self.forward_encoder(
                input_features, attention_mask
            )
            if self.encoder_only:
                return enc_out, enc_lens
            abs_enc_lens = torch.round(enc_lens * enc_out.shape[1])
            logits, _, _ = self.forward_decoder(
                enc_out, decoder_input_ids, abs_enc_lens
            )
            return enc_out, enc_lens, logits

        if self.freeze:
            with torch.no_grad():
                return _forward()
        return _forward()

    def compute_features(self, wav, wav_lens=None):
        """Computes the fbank features expected by the speech encoder with the
        HuggingFace feature extractor (80 mel bins stacked by 2, normalised
        per utterance).

        Arguments
        ---------
        wav : torch.Tensor
            A batch of audio signals, shape (batch, time).
        wav_lens : torch.Tensor
            Relative lengths of the signals in SpeechBrain format.

        Returns
        -------
        input_features : torch.Tensor
            Features, shape (batch, frames, 160).
        attention_mask : torch.Tensor
            Mask of the valid frames, shape (batch, frames).
        """
        # The HuggingFace feature extractor works on numpy arrays, so the
        # features are computed on CPU and moved back to the device.
        wav_cpu = wav.detach().float().cpu()
        if wav_lens is None:
            abs_lens = [wav.shape[1]] * wav.shape[0]
        else:
            abs_lens = torch.round(wav_lens.detach().cpu() * wav.shape[1])
        inputs = self.feature_extractor(
            [w[: int(n)].numpy() for w, n in zip(wav_cpu, abs_lens)],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return (
            inputs["input_features"].to(wav.device),
            inputs["attention_mask"].to(wav.device),
        )

    def forward_encoder(self, input_features, attention_mask=None):
        """Encodes the speech features.

        Arguments
        ---------
        input_features : torch.Tensor
            Features from ``compute_features``, shape (batch, frames, 160).
        attention_mask : torch.Tensor
            Mask of the valid frames, shape (batch, frames).

        Returns
        -------
        enc_out : torch.Tensor
            The speech encoder outputs, shape (batch, enc_frames, hidden).
        enc_lens : torch.Tensor
            Relative lengths of the encoder outputs.
        """
        enc_out = self.model.speech_encoder(
            input_features=input_features, attention_mask=attention_mask
        ).last_hidden_state

        if attention_mask is None:
            return enc_out, enc_out.new_ones(enc_out.shape[0])

        if self.config.add_adapter:
            # Same lengths as the ones HF computes after the length adapter.
            subsample = (
                self.model._compute_sub_sample_lengths_from_attention_mask
            )
            frame_lens = subsample(attention_mask)
        else:
            frame_lens = attention_mask.sum(dim=1)
        enc_lens = (frame_lens.float() / enc_out.shape[1]).clamp(max=1.0)
        return enc_out, enc_lens.to(enc_out.device)

    def forward_decoder(
        self,
        encoder_states,
        decoder_input_ids,
        enc_lens=None,
        last_position_only=False,
        use_cache=False,
        past_key_values=None,
    ):
        """Runs the decoder on a sequence starting with the target language token.

        Arguments
        ---------
        encoder_states : torch.Tensor
            The speech encoder outputs, shape (batch, enc_frames, hidden).
        decoder_input_ids : torch.Tensor
            Decoder inputs starting with the target language token.
        enc_lens : torch.Tensor
            Absolute lengths of the encoder outputs, used to mask padding in
            the cross-attention. If None, nothing is masked.
        last_position_only : bool
            If True, only the last position goes through the output layer,
            as with a 256k-token vocabulary the logits of every position
            take a lot of memory during the search.
        use_cache : bool
            If True, the keys and values are returned for KV caching.
        past_key_values : transformers.Cache
            The key-value cache of the previous steps. If given, only the
            last token of ``decoder_input_ids`` goes through the decoder, and
            only the last position is returned.

        Returns
        -------
        logits : torch.Tensor
            The logits for each input position, shape (batch, tokens, vocab),
            or for the last one, shape (batch, 1, vocab).
        attn : torch.Tensor | None
            The cross-attention weights of the last layer, averaged over the
            heads, if ``output_attentions``: shape (batch, tokens, enc_frames),
            or (batch, enc_frames) for the last position only.
        past_key_values : transformers.Cache | None
            The updated key-value cache if ``use_cache``.
        """
        decoder_input_ids = decoder_input_ids.long()
        if past_key_values is None:
            start = decoder_input_ids.new_full(
                (decoder_input_ids.shape[0], 1), self.decoder_start
            )
            decoder_input_ids = torch.cat([start, decoder_input_ids], dim=1)
        else:
            # The cache holds the previous tokens, only the new one is needed.
            decoder_input_ids = decoder_input_ids[:, -1:]
            last_position_only = True

        encoder_attention_mask = None
        if enc_lens is not None:
            encoder_attention_mask = length_to_mask(
                enc_lens.long(),
                max_len=encoder_states.shape[1],
                dtype=encoder_states.dtype,
                device=encoder_states.device,
            )

        output = self.model.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=self.output_attentions,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        attn = None
        if self.output_attentions:
            # Averaged over heads, as the SpeechBrain searchers expect.
            attn = output.cross_attentions[-1].mean(dim=1)
            attn = attn[:, -1] if last_position_only else attn[:, 1:]

        if last_position_only:
            hidden = output.last_hidden_state[:, -1:]
        else:
            # The first position predicts the language token, which is given.
            hidden = output.last_hidden_state[:, 1:]
        logits = self.model.lm_head(hidden).float()
        return logits, attn, output.past_key_values if use_cache else None
