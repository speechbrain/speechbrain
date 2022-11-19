"""This lobe enables the integration of huggingface pretrained whisper model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Adel Moumen 2022
"""

import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch import nn

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
    >>> model = HuggingFaceWhisper(model_hub, save_path)
    >>> tokens = torch.tensor([[1, 1]]) * model.model.config.decoder_start_token_id
    >>> inputs = torch.rand([1, 93680])
    >>> outputs = model(inputs, tokens)
    """

    def __init__(
        self,
        source,
        save_path,
        sampling_rate,
        freeze=False,
        freeze_feature_extractor=False,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
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
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                logger.warning(
                "speechbrain.lobes.models.huggingface_whisper - whisper encoder is frozen."
                )
                self.model.feature_extractor._freeze_parameters()

    def forward(self, wav, tokens):
        """Perform mel transformation and one step of the whisper (encoder-decoder).

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        tokens : torch.Tensor
            A batch of whisper decoder input ids.
        """
        if self.freeze:
            with torch.no_grad():
                audio_features = self.forward_encoder(wav)
                out = self._get_decoder_hidden_state(audio_features, tokens)
                return out
        else:
            audio_features = self.forward_encoder(wav)
            out = self._get_decoder_hidden_state(audio_features, tokens)
            return out


    def forward_encoder(self, wav):
        """Perform one step of the whisper encoder.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        if self.freeze or self.freeze_feature_extractor:
            with torch.no_grad():
                mel = self._get_mel(wav)
                return self._get_audio_features(mel)
        else:
            mel = self._get_mel(wav)
            return self._get_audio_features(mel)
        
    def _get_audio_features(self, mel):
        """Takes an input mel and return its corresponding whisper encoding.
        Arguments
        ---------
        mel : torch.Tensor
            A batch of mel to transform to features.
        """
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
        return self.feature_extractor(numpy_wav, return_tensors="pt", sampling_rate=self.sampling_rate).input_features.to(wav.device)

    def _get_decoder_hidden_state(self, audio_features, tokens):
        """Perform one step of the whisper decoder.
        Arguments
        ---------
        audio_features : torch.Tensor 
            A batch of audio features (mel + whisper encoding).
        tokens : torch.Tensor
            A batch of whisper decoder input ids.
        """
        return self.model.decoder(encoder_hidden_states=audio_features, input_ids=tokens).last_hidden_state 
