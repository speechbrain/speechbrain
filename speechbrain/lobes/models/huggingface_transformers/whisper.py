"""This lobe enables the integration of huggingface pretrained whisper model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Adel Moumen 2022, 2024
 * Titouan Parcollet 2022
 * Luca Della Libera 2022
 * Ha Nguyen 2023
"""

import torch
import logging
from torch import nn
from functools import cached_property 
import numpy as np 

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)

logger = logging.getLogger(__name__)


class Whisper(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained Whisper model.

    Source paper whisper:
        https://cdn.openai.com/papers/whisper.pdf
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    Some part of the code also cis adapted from the official OpenAI repository:
    https://github.com/openai/whisper

    The model can be finetuned. It will download automatically the model from
    HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "openai/whisper-tiny"
    save_path : str
        Path (dir) of the downloaded model.
    sampling_rate : int (default: 16000)
        Sampling rate of the audio signal.
    encoder_only : bool (default: False)
        If True, the forward function outputs the hidden states from the last transformer layer of the encoder.
        If False, one step of the decoder is performed and returned.
    freeze : bool (default: False)
        If True, the model is frozen.
    freeze_encoder : bool (default: False)
        If True, the encoder is frozen.
    output_attentions : bool (default: True)
        If True, the forward function outputs the attention weights.
    output_all_hiddens: bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers of the encoder.
        For example whisper-base has 6 transformer layers and the output is of shape (7, B, T, C),
        where the output of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer of the encoder.

    Example
    -------
    >>> model_hub = "openai/whisper-tiny"
    >>> save_path = "savedir"
    >>> sampling_rate = 16000
    >>> model = Whisper(model_hub, save_path, sampling_rate)
    >>> tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id
    >>> inputs = torch.randn([1, 93680])
    >>> outputs = model(inputs, tokens)
    """

    def __init__(
        self,
        source,
        save_path,
        sampling_rate=16000,
        encoder_only=False,
        freeze=False,
        freeze_encoder=False,
        output_attentions=False,
        output_all_hiddens=False,
        language="en",
        task="transcribe",
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            sampling_rate=sampling_rate,
        )
        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.freeze_encoder = freeze_encoder
        self.output_attentions = output_attentions
        self.output_all_hiddens = output_all_hiddens
        self.language = language
        self.task = task

        if encoder_only:
            self.tokenizer = None
        else:
            self.load_tokenizer(
                source, 
                bos_token="<|startoftranscript|>", 
                language=self.language,
                task=self.task,
            )

        self.load_feature_extractor(
            source, save_path, sampling_rate=sampling_rate
        )

        self._n_fft = self.feature_extractor.n_fft
        self._hop_length = self.feature_extractor.hop_length
        self._n_samples = self.feature_extractor.n_samples
        # The following breaking changes were introduced in transformers>=4.29:
        # 1) mel_filters.shape = (..., feature_extractor.feature_size) instead of (feature_extractor.feature_size, ...)
        # 2) mel_filters.dtype = float64 instead of float32
        # The following code fixes the issue in a backward compatible way
        mel_filters = self.feature_extractor.mel_filters
        if mel_filters.shape[0] != self.feature_extractor.feature_size:
            mel_filters = mel_filters.T
        assert mel_filters.shape[0] == self.feature_extractor.feature_size
        self.register_buffer(
            "_mel_filters", torch.as_tensor(mel_filters, dtype=torch.float32)
        )
        #################################################################

        if not self.freeze and self.freeze_encoder:
            logger.warning(
                "speechbrain.lobes.models.huggingface_transformers.whisper - whisper encoder is frozen."
            )
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def freeze_model(self, model):
        """
        Freezes parameters of a model.

        Arguments
        ---------
        model : from AutoModel.from_config
            Valid HuggingFace transformers model object.
        """

        logger.warning(
            "speechbrain.lobes.models.huggingface_transformers.whisper - whisper encoder-decoder is frozen."
        )
        model.train()  # we keep it to train to have dropout and LN computed adequaly
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, wav, decoder_input_ids=None):
        """Perform mel transformation and one step of the whisper (encoder-decoder).

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        decoder_input_ids : torch.Tensor
            This is necessary if we want to use the decoder.

            A batch of decoder inputs tokens.
            The first tokens need to dictacte the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.

            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.
        """
        if self.freeze:
            with torch.no_grad():
                out_encoder = self.forward_encoder(wav)
                if self.encoder_only:
                    return out_encoder

                if self.output_all_hiddens:
                    logits, attn, _ = self.forward_decoder(
                        out_encoder[-1], decoder_input_ids
                    )
                else:
                    logits, attn, _ = self.forward_decoder(
                        out_encoder, decoder_input_ids
                    )
                return out_encoder, logits, attn
        else:
            if self.encoder_only:
                return self.forward_encoder(wav)
            else:
                out_encoder = self.forward_encoder(wav)
                if self.output_all_hiddens:
                    logits, attn, _ = self.forward_decoder(
                        out_encoder[-1], decoder_input_ids
                    )
                else:
                    logits, attn, _ = self.forward_decoder(
                        out_encoder, decoder_input_ids
                    )
                return out_encoder, logits, attn

    def forward_encoder(self, wav):
        """Perform one step of the whisper encoder with Mel FBANKs as Input.

        Arguments
        ---------
        wav : torch.Tensor (FBANKs)
            A batch of Mel FBANK from HF to transform to features.
        """

        if self.freeze_encoder:
            with torch.no_grad():
                return self._get_encoder_states(wav)
        else:
            return self._get_encoder_states(wav)

    def _get_encoder_states(self, wav):
        """Takes an input waveform and return its corresponding encoder states.
        Returns the last hidden state of the encoder or all hidden states if
        output_all_hiddens is True.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        mel = self._get_mel(wav)
        if self.output_all_hiddens:
            states = self.model.encoder(mel, output_hidden_states=True)
            return torch.stack(states.hidden_states)
        else:
            return self.model.encoder(mel).last_hidden_state

    def _get_mel(self, wav):
        """Takes an input waveform and return its corresponding mel spectrogram
        according to HuggingFace implementation. WARNING: it's slow! Better push this
        in the DataLoader.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        mels = self._pad_or_trim(wav)
        mels = self._log_mel_spectrogram(mels)
        return mels

    def _log_mel_spectrogram(self, audio):
        """Compute the Mel spectrogram of a batch of input waveforms.

        Reference: adapted from
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L92

        Arguments
        ---------
        audio : torch.Tensor
            A batch of audio waveforms in 16 kHz.

        Returns
        -------
        torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        """
        window = torch.hann_window(self._n_fft, device=audio.device)
        stft = torch.stft(
            audio,
            self._n_fft,
            self._hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self._mel_filters
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(
            log_spec,
            log_spec.max() - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _pad_or_trim(self, array, axis=-1):
        """Pad or trim the Mel spectrograms as expected by the encoder.

        Reference: adapted from
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L52

        Arguments
        ---------
        array : torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        axis : int
            The axis along which to pad.

        Returns
        -------
        torch.Tensor
            The padded tensor.
        """
        if array.shape[axis] > self._n_samples:
            array = array.index_select(
                dim=axis,
                index=torch.arange(self._n_samples, device=array.device),
            )

        if array.shape[axis] < self._n_samples:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (
                0,
                self._n_samples - array.shape[axis],
            )
            array = nn.functional.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )

        return array

    def forward_decoder(self, audio_features, decoder_input_ids, use_cache=True, past_key_values=None):
        """Perform one step of the whisper decoder.

        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (mel + whisper encoding).
        decoder_input_ids : torch.Tensor
            A batch of decoder inputs tokens.
            The first tokens need to dictacte the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.

            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.
        """
        # print(decoder_input_ids)
        if past_key_values is not None:
            # print(decoder_input_ids.shape)
            decoder_input_ids = decoder_input_ids[:, -1].unsqueeze(-1)
            # print(decoder_input_ids.shape)
        output_states = self.model.decoder(
            encoder_hidden_states=audio_features,
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            output_attentions=self.output_attentions,
            use_cache=use_cache,
        )
        # print(len(output_states.past_key_values))
        # exit()
        # output_states = output_states.last_hidden_state
        # print(output_states.last_hidden_state.shape)
        logits = output_states.last_hidden_state @ self.model.decoder.embed_tokens.weight.T
        # print(logits.shape)
        return logits, None, output_states.past_key_values

    @cached_property
    def all_language_tokens(self):
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        langs = list(LANGUAGES.keys())  # Convert keys to a list
        bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        result = []
        for lang in langs:
            result.append(
                bos_token_id + 1 + langs.index(lang)
            )
        return tuple(result)

    @cached_property
    def all_language_codes(self):
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        langs = list(LANGUAGES.keys())  # Convert keys to a list
        # bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        return tuple(langs)

    @cached_property
    def non_speech_tokens(self):
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.tokenizer.encode(" -", add_special_tokens=False)[0], self.tokenizer.encode(" '", add_special_tokens=False)[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.tokenizer.encode(symbol, add_special_tokens=False),
                self.tokenizer.encode(" " + symbol, add_special_tokens=False),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    @cached_property
    def transcribe(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|transcribe|>")

    @cached_property
    def translate(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|translate|>")

    @cached_property
    def bos(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    
    @cached_property
    def eos(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    @cached_property
    def bos_lm(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|startoflm|>")

    @cached_property
    def bos_prev(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|startofprev|>")

    @cached_property
    def no_speech(self) -> int:
        # TODO: inspect this. for a given reason nospeech maps to eos...
        return 50362 # self.tokenizer.convert_tokens_to_ids("<|nospeech|>") 

    @cached_property
    def no_timestamps(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    @cached_property
    def timestamp_begin(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|0.00|>")

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.tokenizer.convert_tokens_to_ids.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @torch.no_grad()
    def detect_language(self, mel):
        if self.tokenizer.language is None:
            raise ValueError(
                "This model doesn't have language tokens so it can't perform lang id"
            )
        
        # forward pass using a single token, startoftranscript
        n_audio = mel.shape[0]

        audio_features = self.model.encoder(mel).last_hidden_state

        bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        decoder_input_ids = torch.tensor([[bos_token_id]] * n_audio).to(mel.device)  # [n_audio, 1]
        # print(decoder_input_ids)
        logits = self.forward_decoder(audio_features, decoder_input_ids)[0][:, 0]
        # print(logits)
        # collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(self.all_language_tokens)] = False
        # print(mask)
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        # print("tokens = ", self.all_language_tokens)
        # print("codes = ", self.all_language_codes)
        # exit()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(self.all_language_tokens, self.all_language_codes)
            }
            for i in range(n_audio)
        ]
        
        if mel.shape[0] == 1:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        return language_tokens, language_probs

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)