import logging

from torch import Tensor
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

logger = logging.getLogger(__name__)


class HuggingFaceGPT(nn.Module):
    """This lobe enables the integration of HuggingFace pretrained GPT model.
      Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "gpt2"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    """

    def __init__(self, source: str, save_path: str, freeze: bool = False) -> None:
        super().__init__()
        self.freeze = freeze
        self.model = GPT2LMHeadModel.from_pretrained(source, cache_dir=save_path)
        if self.freeze:
            logger.warning(
                "huggingface_GPT - GPT  is frozen."
            )
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, input_ids: Tensor, token_type_ids: Tensor):
        """ Takes an input  and return its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor ()
            A batch of input-id to transform to features.
        token_type_ids : torch.Tensor
            This is necessary if we want to use the decoder.
        """
        
        with torch.set_grad_enabled(not self.freeze):
            output = self.model.forward(input_ids, token_type_ids=token_type_ids)
        return output
