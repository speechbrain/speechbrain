"""This lobe enables the integration of huggingface pretrained whisper model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Adel Moumen 2022
 * Titouan Parcollet 2022
 * Luca Della Libera 2022
"""

import torch
import logging
from torch import nn

try:
    from transformers import WhisperModel
    from transformers import WhisperFeatureExtractor
    from transformers.models.whisper.tokenization_whisper import (
        WhisperTokenizer,
    )
except ImportError:
    MSG = "Please install transformers from HuggingFace to use Whisper\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class HuggingFaceWhisper(nn.Module):
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
    >>> model = HuggingFaceWhisper(model_hub, save_path, sampling_rate)
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
        output_attentions=True,
        output_all_hiddens=False,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.freeze = freeze
        self.freeze_encoder = freeze_encoder
        self.output_attentions = output_attentions
        self.output_all_hiddens = output_all_hiddens

        self.tokenizer = None
        # Download the tokenizer only if we are going to use the Decoder.
        if not encoder_only:
            self.tokenizer = WhisperTokenizer.from_pretrained(source)

        # Download the extractor from HuggingFace.
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            source, cache_dir=save_path, sampling_rate=sampling_rate,
        )
        self._n_fft = feature_extractor.n_fft
        self._hop_length = feature_extractor.hop_length
        self._n_samples = feature_extractor.n_samples
        # The following breaking changes were introduced in transformers>=4.29:
        # 1) mel_filters.shape = (..., feature_extractor.feature_size) instead of (feature_extractor.feature_size, ...)
        # 2) mel_filters.dtype = float64 instead of float32
        # The following code fixes the issue in a backward compatible way
        mel_filters = feature_extractor.mel_filters
        if mel_filters.shape[0] != feature_extractor.feature_size:
            mel_filters = mel_filters.T
        assert mel_filters.shape[0] == feature_extractor.feature_size
        self.register_buffer(
            "_mel_filters", torch.as_tensor(mel_filters, dtype=torch.float32)
        )
        #################################################################

        self.model = WhisperModel.from_pretrained(source, cache_dir=save_path)

        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_whisper - whisper encoder-decoder is frozen."
            )
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_encoder:
                logger.warning(
                    "speechbrain.lobes.models.huggingface_whisper - whisper encoder is frozen."
                )
                for param in self.model.encoder.parameters():
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
                    logits, attn = self.forward_decoder(
                        out_encoder[-1], decoder_input_ids
                    )
                else:
                    logits, attn = self.forward_decoder(
                        out_encoder, decoder_input_ids
                    )
                return out_encoder, logits, attn
        else:
            if self.encoder_only:
                return self.forward_encoder(wav)
            else:
                out_encoder = self.forward_encoder(wav)
                if self.output_all_hiddens:
                    logits, attn = self.forward_decoder(
                        out_encoder[-1], decoder_input_ids
                    )
                else:
                    logits, attn = self.forward_decoder(
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
            (log_spec.flatten(start_dim=1).max(dim=-1)[0] - 8.0)[:, None, None],
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

    def forward_decoder(self, audio_features, decoder_input_ids):
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
        output_states = self.model.decoder(
            encoder_hidden_states=audio_features,
            input_ids=decoder_input_ids,
            output_attentions=self.output_attentions,
        )

        attn = output_states.attentions[-1]
        attn = attn.view(attn.shape[0] * attn.shape[1], *attn.shape[2:])
        output_states = output_states.last_hidden_state

        logits = (
            output_states
            @ torch.transpose(
                self.model.decoder.embed_tokens.weight.to(output_states.dtype),
                0,
                1,
            )
        ).to(audio_features.dtype)

        return logits, attn
