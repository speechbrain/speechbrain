"""This lobe enables the integration of huggingface pretrained GPT2LMHeadModel model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Pooneh Mousavi 2023
 * Simone Alghisi 2023
"""

import torch

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class GPT(HFTransformersInterface):
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
    max_new_tokens : int
        Maximum count of new tokens allowed.
    min_length : int
        Minimum count of input tokens
    top_k : int
        Top results count to keep
    top_p : float
        Proportion of top results to keep
    num_beams : int
        Number of decoder beams
    eos_token_id : int
        Index of end-of-sentence token.
    early_stopping : int
        Whether to stop training early.

    Example
    -------
    >>> model_hub = "gpt2"
    >>> save_path = "savedir"
    >>> model = GPT(model_hub, save_path)
    >>> tokens = torch.tensor([[1, 1]])
    >>> tokens_type = torch.tensor([[1, 1]])
    >>> attention_mask = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, tokens_type, attention_mask)
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=False,
        max_new_tokens=200,
        min_length=1,
        top_k=45,
        top_p=0.9,
        num_beams=8,
        eos_token_id=50258,
        early_stopping=True,
    ) -> None:
        super().__init__(
            source=source, save_path=save_path, freeze=freeze, with_lm_head=True
        )
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.eos_token_id = eos_token_id

        self.load_tokenizer(source=source, pad_token=None, use_fast=False)

        if self.freeze:
            logger.warning("huggingface_GPT - GPT  is frozen.")
            self.model.train()  # we keep it to train to have dropout and LN computed adequately
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor
            A batch of input-id to transform to features.
        token_type_ids : torch.Tensor
            Token Type(Speaker) for each token in input_ids.
        attention_mask : torch.Tensor
            A batch of attention_mask.

        Returns
        -------
        output : torch.Tensor
            Reply to conversation
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
        input_ids: torch.Tensor,
        token_type_ids,
        attention_mask: torch.Tensor,
        decoder_type="greedy",
    ):
        """Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor
            A batch of input-id which are dialogue context tokens
        token_type_ids : torch.Tensor
        attention_mask : torch.Tensor
            A batch of attention_mask.
        decoder_type : str
            It shows strategy for autoregressive decoding either beam search or greedy.

        Returns
        -------
        hyp : torch.Tensor
            Conversation reply.
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
                    eos_token_id=self.eos_token_id,
                    early_stopping=self.early_stopping,
                )
            else:
                # greedy decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    input_ids,
                    token_type_ids=token_type_ids,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.eos_token_id,
                    attention_mask=attention_mask,
                )
        return hyp
