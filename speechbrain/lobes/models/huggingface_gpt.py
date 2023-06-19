"""This lobe enables the integration of huggingface pretrained GPT2LMHeadModel model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Pooneh Mousavi 2023
 * Simone Alghisi 2023
"""

import logging
from torch import Tensor
import torch
import torch.nn as nn

try:
    from transformers import GPT2LMHeadModel
except ImportError:
    MSG = "Please install transformers from HuggingFace to use GPT2\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class HuggingFaceGPT(nn.Module):
    """This lobe enables the integration of HuggingFace pretrained GPT model.
     Source paper whisper:
        https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf
    Transformer from HuggingFace needs to be installed:
        https://huggingface.co/transformers/installation.html

    The model can be finetuned. It will download automatically the model from
    HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "gpt2"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    Example
    -------
    >>> model_hub = "gpt2"
    >>> save_path = "savedir"
    >>> model = HuggingFaceGPT(model_hub, save_path)
    >>> tokens = torch.tensor([[1, 1]])
    >>> tokens_type = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, tokens_type)
    """

    def __init__(
        self, source: str, save_path: str, freeze: bool = False
    ) -> None:
        super().__init__()
        self.freeze = freeze
        self.model = GPT2LMHeadModel.from_pretrained(
            source, cache_dir=save_path
        )
        if self.freeze:
            logger.warning("huggingface_GPT - GPT  is frozen.")
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, token_type_ids: Tensor):
        """ Takes an input a history of conversation and return its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor ()
            A batch of input-id to transform to features.
        token_type_ids : torch.Tensor
            Token Type(Speaker) for each token in input_ids.
        """

        with torch.set_grad_enabled(not self.freeze):
            output = self.model.forward(
                input_ids, token_type_ids=token_type_ids
            )
        return output
