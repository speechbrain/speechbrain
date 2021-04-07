"""This lobes enables the integration of huggingface pretrained wav2vec2 models.

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
except ImportError:
    print("Please install transformer from HuggingFace to use wav2vec2!")


class HuggingFaceWav2Vec2(nn.Module):
    """This lobes enables the integration of HuggingFace
    pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed features extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    pretrained_path : str
        Path of the pretrained wav2vec2 model. It can be a url or a local path.
    save_path : str
        Path and filename of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.

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

    def __init__(self, source, save_path, output_norm=True, freeze=True):
        super().__init__()

        # Download the model from HuggingFace and load it.
        self.model = Wav2Vec2Model.from_pretrained(source, cache_dir=save_path)
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

        # Extract wav2vec output
        out = self.model(wav)[0]
        print(out.shape)

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out
