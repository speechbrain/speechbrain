"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
"""

import torch
import torch.nn.functional as F
from torch import nn

# We check if transformers is installed.
try:
    from transformers import Wav2Vec2Model
    from transformers import Wav2Vec2Processor
except ImportError:
    print("Please install transformer from HuggingFace to use wav2vec2!")


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self, source, save_path, output_norm=True, freeze=True, pretrain=True
    ):
        super().__init__()

        # Download the model from HuggingFace and load it.
        # The Processor is only used to retrieve the normalisation
        self.proc = Wav2Vec2Processor.from_pretrained(
            source, cache_dir=save_path
        )
        self.model = Wav2Vec2Model.from_pretrained(source, cache_dir=save_path)

        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            self.reset_layer(self.model)

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.proc.feature_extractor.do_normalize

        self.freeze = freeze
        self.output_norm = output_norm
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out = self.model(wav)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)
