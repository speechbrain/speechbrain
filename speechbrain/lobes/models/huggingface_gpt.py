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
    >>> attention_mask = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, tokens_type, attention_mask)
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        freeze: bool = False,
        max_new_tokens: int = 200,
        min_length: int = 1,
        top_k: int = 45,
        top_p: float = 0.9,
        num_beams: int = 8,
        early_stopping: bool = True,
    ) -> None:
        super().__init__()
        self.freeze = freeze
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.model = GPT2LMHeadModel.from_pretrained(
            source, cache_dir=save_path
        )
        if self.freeze:
            logger.warning("huggingface_GPT - GPT  is frozen.")
            self.model.train()  # we keep it to train to have dropout and LN computed adequaly
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor,
    ):
        """ Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor ()
            A batch of input-id to transform to features.
        token_type_ids : torch.Tensor
            Token Type(Speaker) for each token in input_ids.
        attention_mask : torch.Tensor ()
            A batch of attention_mask.
        """
        with torch.set_grad_enabled(not self.freeze):
            output = self.model.forward(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        return output

    def generate(
        self,
        input_ids: Tensor,
        token_type_ids,
        attention_mask: Tensor,
        decoder_type="greedy",
    ):
        """ Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        --------
        input_ids : torch.Tensor ()
            A batch of input-id   which are dialogue context tokens
        decoder_type : Str
            It shows strategy for autoregressive decoding either beam seach or greedy.
        attention_mask : torch.Tensor ()
            A batch of attention_mask.
        """

        with torch.no_grad():
            if decoder_type == "beam":
                # beam decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=self.max_new_tokens,
                    min_length=self.min_length,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    num_return_sequences=1,
                    # pad_token_id=50258,
                    eos_token_id=50258,
                    early_stopping=self.early_stopping,
                )
            else:
                # greedy decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    input_ids,
                    token_type_ids=token_type_ids,
                    max_new_tokens=self.max_new_tokens,
                    # pad_token_id=50258,
                    eos_token_id=50258,
                    attention_mask=attention_mask,
                )
        return hyp
