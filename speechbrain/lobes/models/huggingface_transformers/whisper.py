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
            # when the model is not multilingual i.e. all Whisper
            # models ending in .en, you must not set the language
            # and task tokens.
            self.load_tokenizer(
                source, bos_token="<|startoftranscript|>",
            )

            if self.is_multilingual:
                self.tokenizer.set_prefix_tokens(
                    language=self.language, task=self.task
                )

        self.load_feature_extractor(
            source, save_path, sampling_rate=sampling_rate
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
        model.train()  # we keep it to train to have dropout and LN computed adequately
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
            The first tokens need to dictate the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.

            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.
        """

        def _fwd():
            out_encoder = self.forward_encoder(wav)
            if self.encoder_only:
                return out_encoder
            else:
                if self.output_all_hiddens:
                    decoder_logits, decoder_attn, _ = self.forward_decoder(
                        out_encoder[-1], decoder_input_ids
                    )
                else:
                    decoder_logits, decoder_attn, _ = self.forward_decoder(
                        out_encoder, decoder_input_ids
                    )
                return out_encoder, decoder_logits, decoder_attn

        if self.freeze:
            with torch.no_grad():
                return _fwd()
        else:
            return _fwd()

    def forward_encoder(self, wav):
        """Perform one step of the whisper encoder with Mel FBANKs as Input.

        Arguments
        ---------
        wav : torch.Tensor (FBANKs)
            A batch of Mel FBANK from HF to transform to features.
        """
        return self._get_encoder_states(wav)

    def _get_mel(self, wav):
        mel = torch.from_numpy(
            np.array(
                self.feature_extractor(
                    wav.cpu().numpy(), sampling_rate=self.sampling_rate
                ).input_features
            ),
        ).to(self.model.device)
        return mel

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

    def forward_decoder(
        self,
        audio_features,
        decoder_input_ids,
        use_cache=True,
        past_key_values=None,
    ):
        """Perform one step of the whisper decoder.

        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (mel + whisper encoding).
        decoder_input_ids : torch.Tensor
            A batch of decoder inputs tokens.
            The first tokens need to dictate the behavior of the decoder.
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
        # logits = output_states.last_hidden_state @ self.model.decoder.embed_tokens.weight.T

        logits = (
            output_states.last_hidden_state
            @ self.model.decoder.embed_tokens.weight.T
        )

        # print(logits.shape)
        return logits, None, output_states.past_key_values

    @cached_property
    def all_language_tokens(self):
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        langs = list(LANGUAGES.keys())  # Convert keys to a list
        bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.bos_token
        )
        result = []
        for lang in langs:
            result.append(bos_token_id + 1 + langs.index(lang))
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
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {
            self.tokenizer.encode(" -", add_special_tokens=False)[0],
            self.tokenizer.encode(" '", add_special_tokens=False)[0],
        }
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
        return 50362  # self.tokenizer.convert_tokens_to_ids("<|nospeech|>")

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
            raise ValueError(
                "This tokenizer does not have language token configured"
            )
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        token = self.tokenizer.convert_tokens_to_ids.get(
            f"<|{language}|>", None
        )
        if token:
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    def set_language_token(self, language):
        self.language = language
        self.tokenizer.set_prefix_tokens(language=self.language)

    def set_task(self, task):
        self.task = task
        self.tokenizer.set_prefix_tokens(task=self.task)

    @cached_property
    def is_multilingual(self):
        return len(self.tokenizer) >= 51865

    @torch.no_grad()
    def detect_language(self, mel):
        if self.tokenizer.language is None:
            raise ValueError(
                "This model doesn't have language tokens so it can't perform lang id"
            )

        # forward pass using a single token, startoftranscript
        n_audio = mel.shape[0]

        audio_features = self.model.encoder(mel).last_hidden_state

        decoder_input_ids = torch.tensor([[self.bos]] * n_audio).to(
            mel.device
        )  # [n_audio, 1]
        # print(decoder_input_ids)
        logits = self.forward_decoder(audio_features, decoder_input_ids)[0][
            :, 0
        ]
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
                for j, c in zip(
                    self.all_language_tokens, self.all_language_codes
                )
            }
            for i in range(n_audio)
        ]

        if mel.shape[0] == 1:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        return language_tokens, language_probs
