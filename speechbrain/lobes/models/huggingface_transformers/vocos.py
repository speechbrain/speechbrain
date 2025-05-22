"""This lobe enables the integration of huggingface pretrained
Vocos model.

Vocos is a vocoder trained on top of EnCodec tokens. While
EnCodec itself can be used for a lossy reconstruction of speech,
a vocoder, such as Vocos, can be used to improve the quality.

Repository: https://huggingface.co/charactr/vocos-encodec-24khz
Paper: https://arxiv.org/pdf/2306.00814.pdf

TODO: There is an open feature request to add this model to
HuggingFace Transformers.

If this is implemented, it will be possible to make this model
inherit from HFTransformersInterface

https://github.com/huggingface/transformers/issues/25123

Authors
 * Artem Ploujnikov 2023
"""

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.logger import get_logger

try:
    from vocos import Vocos as VocosModel
    from vocos.feature_extractors import EncodecFeatures
except ImportError:
    MSG = "Please install vocos to use the Vocos model\n"
    MSG += "E.G. run: pip install vocos"
    raise ImportError(MSG)


DEFAULT_SAMPLE_RATE = 24000
BANDWIDTHS = [1.5, 3.0, 6.0, 12.0]

logger = get_logger(__name__)


# cspell:ignore charactr
class Vocos(nn.Module):
    """An wrapper for the HuggingFace Vocos model

    Arguments
    ---------
    source : str
        A HuggingFace repository identifier or a path
    save_path : str
        The location where the pretrained model will be saved
    revision : str
        The model revision
    bandwidth : float
        The bandwidth value
        Supported:
        1.5, 3.0, 6.0, 12.0
    freeze : bool
        Whether or not parameters should be
        frozen

    Example
    -------
    >>> model_hub = "charactr/vocos-encodec-24khz"
    >>> save_path = "savedir"
    >>> model = Vocos(model_hub, save_path)
    >>> tokens = torch.randint(1024, (4, 10, 2))
    >>> length = torch.tensor([1.0, 0.5, 0.75, 1.0])
    >>> audio, out_length = model(tokens, length)
    >>> audio.shape
    torch.Size([4, 3200])
    >>> out_length
    tensor([1.0000, 0.5000, 0.7500, 1.0000])
    """

    def __init__(
        self,
        source,
        save_path,
        revision=None,
        bandwidth=1.5,
        freeze=True,
    ):
        super().__init__()
        self.source = source
        self.save_path = save_path
        self.revision = revision
        self.model = self._load_model()
        self.freeze = freeze
        self.bandwidth = bandwidth
        self.bandwidth_id = (
            (torch.tensor(BANDWIDTHS) - bandwidth).abs().argmin().item()
        )
        if self.freeze:
            logger.warning("huggingface_Vocos - Vocos is frozen.")
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_model(self):
        """Loads the pretrained model. This is a customized implementation of
        Vocos.from_pretrained(), which has been customized to specify an
        alternate cache_dir"""
        config_path = hf_hub_download(
            repo_id=self.source,
            filename="config.yaml",
            revision=self.revision,
            cache_dir=self.save_path,
        )
        model_path = hf_hub_download(
            repo_id=self.source,
            filename="pytorch_model.bin",
            revision=self.revision,
            cache_dir=self.save_path,
        )
        model = VocosModel.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, inputs, length):
        """Converts EnCodec tokens to audio

        Arguments
        ---------
        inputs : torch.Tensor
            A tensor of EnCodec tokens
        length : torch.Tensor
            A 1-D tensor of relative lengths

        Returns
        -------
        wavs : torch.Tensor
            A (Batch x Length) tensor of raw waveforms
        length : torch.Tensor
            Relative lengths
        """
        with torch.set_grad_enabled(not self.freeze):
            features = self.model.codes_to_features(inputs.permute(2, 0, 1))
            wavs = self.model.decode(
                features,
                bandwidth_id=torch.tensor(
                    [self.bandwidth_id], device=inputs.device
                ),
            )
            mask = length_to_mask(
                length * wavs.size(1), max_len=wavs.size(1), device=wavs.device
            )
            return wavs * mask, length
