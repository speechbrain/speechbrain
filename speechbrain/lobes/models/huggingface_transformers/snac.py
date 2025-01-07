"""This lobe enables the integration of huggingface pretrained
SNAC model.

SNAC is a Multi-Scale Neural Audio Codec (SNAC) compresses audio
into discrete codes at a low bitrate.

Repository: https://github.com/hubertsiuzdak/snac
Website: https://hubertsiuzdak.github.io/snac/

Authors
 * Julien Blanchon 2024
"""

import logging
from typing import List, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from torch import nn

try:
    from snac import SNAC as _SNAC
except ImportError:
    MSG = "Please install snac to use the SNAC model\n"
    MSG += "E.G. run: pip install snac"
    raise ImportError(MSG)


logger = logging.getLogger(__name__)


# cspell:ignore charactr
class SNAC(nn.Module):
    """An wrapper for the HuggingFace SNAC model

    Arguments
    ---------
    source : str
        A HuggingFace repository identifier or a path
    save_path : str
        The location where the pretrained model will be saved
    revision : str
        The model revision
    freeze : bool
        Whether or not parameters should be
        frozen

    Example
    -------
    >>> from speechbrain.lobes.models.discrete.snac import SNAC
    >>> model = SNAC("hubertsiuzdak/snac_32khz", "savedir")
    >>> audio = torch.randn(1, 1, 32000)
    >>> # audio_hat, codes = model(audio)
    >>> codes = model.encode(audio)
    >>> audio_hat = model.decode(codes)
    >>> print([code.shape[1] for code in codes])
    >>> [12, 24, 48, 96]
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        revision: Union[str, None] = None,
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.save_path = save_path
        self.revision = revision
        self.freeze = freeze
        self.kwargs = kwargs
        self.model = self._load_model()
        if self.freeze:
            logger.warning("huggingface_SNAC - SNAC is frozen.")
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self):
        """Loads the pretrained model. This is a customized implementation of
        SNAC.from_pretrained(), which has been customized to specify an
        alternate cache_dir"""
        config_path = hf_hub_download(
            repo_id=self.source,
            filename="config.json",
            revision=self.revision,
            cache_dir=self.save_path,
            **self.kwargs,
        )
        model_path = hf_hub_download(
            repo_id=self.source,
            filename="pytorch_model.bin",
            revision=self.revision,
            cache_dir=self.save_path,
            **self.kwargs,
        )
        model = _SNAC.from_config(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(
        self, audio_data: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Converts SNAC tokens to audio

        Arguments
        ---------
        audio_data : torch.Tensor
            A (Batch x Channels x Length) tensor of audio data

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            A tuple of audio and codes
        """
        audio_hat, codes = self.model(audio_data)
        return audio_hat, codes

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        """Encodes audio data into codes

        Arguments
        ---------
        audio_data : torch.Tensor
            Note that codes is a list of token sequences of variable lengths, each corresponding to a different temporal resolution
            A (Batch x Channels x Length) tensor of audio data

        Returns
        -------
        List[torch.Tensor]
            A list of codes
        """
        return self.model.encode(audio_data)

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """Decodes codes into audio data

        Arguments
        ---------
        codes : List[torch.Tensor]
            A list of codes

        Returns
        -------
        torch.Tensor
            A (Batch x Channels x Length) tensor of audio data
        """
        return self.model.decode(codes)

    def preprocess(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Preprocesses audio data, add padding to match the model's hop length.

        Arguments
        ---------
        audio_data : torch.Tensor
            A (Batch x Channels x Length) tensor of audio data

        Returns
        -------
        torch.Tensor
            A (Batch x Channels x Length) tensor of audio data with padding
        """
        return self.model.preprocess(audio_data)
