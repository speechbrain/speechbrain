"""This lobe enables the integration of huggingface pretrained EnCodec.

EnCodec makes it possible to compress audio into a sequence of discrete tokens
at different bandwidths - and to reconstruct audio from such sequences, with
some loss of quality depending on the bandwidth.

Note that while encodec can be used to reconstruct speech data, for a
high-quality reconstruction, it is recommended to use a specially trained
vocoder, such as Vocos (speechbrain.lobes.models.huggingface_transformers.vocos)

Repository: https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/encodec
Paper: https://arxiv.org/abs/2210.13438

Authors
 * Artem Ploujnikov 2023
"""

import torch
import logging
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)


try:
    from transformers import EncodecModel
except ImportError:
    MSG = "Please install transformers from HuggingFace to use Encodec\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)


DEFAULT_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)


class Encodec(HFTransformersInterface):
    """An wrapper for the HuggingFace encodec model

    Arguments
    ---------
    source : str
        a HuggingFace repository identifier or a path
    save_path : str
        the location where the pretrained model will be saved
    sample_rate : int
        the audio sampling rate
    bandwidth : float
        the encoding bandwidth, in kbps (optional)
        Supported bandwidths:
        1.5, 3.0, 6.0, 12.0, 24.0
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used
        as part of training another model)

    Example
    -------
    >>> model_hub = "facebook/encodec_24khz"
    >>> save_path = "savedir"
    >>> model = Encodec(model_hub, save_path)
    >>> audio = torch.randn(4, 1000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens = model.encode(audio, length)
    >>> tokens.shape
    torch.Size([4, 4, 2])
    >>> rec = model.decode(tokens, length)
    >>>
    """

    def __init__(
        self,
        source,
        save_path=None,
        sample_rate=None,
        freeze=True,
        bandwidth=1.5,
    ):
        super().__init__(source=source, save_path=save_path, freeze=freeze)
        self.model = EncodecModel.from_pretrained(source, cache_dir=save_path)
        if not sample_rate:
            sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        if self.freeze:
            logger.warning("huggingface_Encodec - Encodec is frozen.")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, inputs, length):
        """Encodes the input audio as tokens

        Arguments
        ---------
        inputs : torch.Tensor
            a (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            a (Batch X Tokens) tensor of audio tokens
        """
        return self.encode(inputs, length)

    def encode(self, inputs, length):
        """Encodes the input audio as tokens

        Arguments
        ---------
        inputs : torch.Tensor
            a (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            a (Batch x Tokens x Heads) tensor of audio tokens
        """
        with torch.set_grad_enabled(not self.freeze):
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
            max_len = inputs.size(-1)
            mask = length_to_mask(
                length * max_len, max_len, device=inputs.device
            ).unsqueeze(1)
            result = self.model.encode(inputs, mask, bandwidth=self.bandwidth)
            return result.audio_codes.squeeze(0).transpose(-1, -2)

    def decode(self, tokens, length=None):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            a tensor of tokens
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        audio : torch.Tensor
            the audio
        """
        with torch.set_grad_enabled(not self.freeze):
            result = self.model.decode(
                tokens.unsqueeze(0).transpose(-1, -2), [None]
            )
            audio = result.audio_values
            if length is not None:
                max_len = audio.size(-1)
                mask = length_to_mask(
                    length * max_len, max_len, device=tokens.device
                ).unsqueeze(1)
                audio = audio * mask
            return audio
