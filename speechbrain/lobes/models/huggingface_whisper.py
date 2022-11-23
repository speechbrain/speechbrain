"""This lobe enables the integration of huggingface pretrained whisper model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Adel Moumen 2022
 * Titouan Parcollet 2022
"""

import torch
import logging
from torch import nn
import torch.nn.functional as F

try:
    from transformers import WhisperModel
    from transformers import WhisperFeatureExtractor

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

    The model can be finetuned. It will download automatically the model from
    HuggingFace or use a local path.
    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "openai/whisper-tiny"
    save_path : str
        Path (dir) of the downloaded model.
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
        freeze_feature_extractor=False,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.freeze = freeze

        # Download the extractor from HuggingFace.
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        self.model = WhisperModel.from_pretrained(source, cache_dir=save_path)
        self.freeze_feature_extractor = freeze_feature_extractor
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_whisper - whisper encoder-decoder is frozen."
            )
            self.model.train() # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                logger.warning(
                    "speechbrain.lobes.models.huggingface_whisper - whisper encoder is frozen."
                )
                for param in self.model.feature_extractor.parameters():
                    param.requires_grad = False

    def forward(self, wav, tokens=None):
        """Perform mel transformation and one step of the whisper (encoder-decoder).

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        tokens : torch.Tensor
            A batch of whisper decoder input ids. This is only necessary if the decoder is used.
        """
        if self.freeze:
            with torch.no_grad():
                out_encoder = self.forward_encoder(wav)
                if self.encoder_only:
                    return out_encoder
                out = self.forward_decoder(out_encoder, tokens)
                return out
        else:
            out_encoder = self.forward_encoder(wav)
            if self.encoder_only:
                return out_encoder
            out = self.forward_decoder(out_encoder, tokens)
            return out

    def forward_encoder(self, wav):
        """Perform one step of the whisper encoder.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.freeze_feature_extractor:
            with torch.no_grad():
                mel = self._get_mel(wav)
                return self.model.encoder(mel).last_hidden_state
        else:
            mel = self._get_mel(wav)
            return self.model.encoder(mel).last_hidden_state

    def _get_mel(self, wav):
        """Takes an input waveform and return its corresponding mel spectrogram.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        # need to cast tensor to numpy for the huggingface whisper feature extractor.
        numpy_wav = wav.cpu().numpy().tolist()
        return self.feature_extractor(
            numpy_wav, return_tensors="pt", sampling_rate=self.sampling_rate
        ).input_features.to(wav.device)

    def forward_decoder(self, audio_features, tokens):
        """Perform one step of the whisper decoder.
        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (mel + whisper encoding).
        tokens : torch.Tensor
            A batch of whisper decoder input ids.
        """
        return self.model.decoder(
            encoder_hidden_states=audio_features, input_ids=tokens
        ).last_hidden_state
