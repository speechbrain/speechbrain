"""This lobe enables the integration of huggingface pretrained LlaMA models.

Authors
 * Titouan Parcollet 2025
 * Shucong Zhang 2025
 * Pooneh Mousavi 2023
"""

from typing import Literal

import torch
from transformers import BitsAndBytesConfig

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class LLaMA(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained LLaMA models.

    The model can be finetuned entirely or coupled with SpeechBrain (and peft) adapters (see https://speechbrain.readthedocs.io/en/latest/tutorials/nn/neural-network-adapters.html)

    Quantisation can be applied by passing a BitsAndBytesConfig which can be instantiated in a SpeechBrain yaml (or elsewhere.)

    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "meta-llama/Llama-2-7b-chat-hf"
    save_path : str
        Path (dir) of the downloaded model.
    bnb_config : transformers.BitsAndBytesConfig
        BitsAndBytesConfig enabling quantisation of the model. If not specified, the model weights will be loaded with weight_precision_load dtype.
    freeze : bool (default: false)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pad_token : str (default: "PAD")
        String representation of the padding token. This may change from one model to another.
    padding_side : str (default: "right")
        One of right or left.
    weight_precision_load : torch.dtype (default: torch.bfloat16)
        If no bnb_config is given, this parameter defines the loading type of the parameters of the model. This is useful to reduce memory footprint, but it does not change the compute dtype. For this just refer to mixed precision training in SpeechBrain.

    Example
    -------
    >>> model_hub = "meta-llama/Llama-2-7b-chat-hf"
    >>> save_path = "savedir"
    >>> model = LLaMA(model_hub, save_path)
    >>> tokens = torch.tensor([[1, 1]])
    >>> attention_mask = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, attention_mask)
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        bnb_config: BitsAndBytesConfig = None,
        freeze: bool = False,
        pad_token: str = "[PAD]",
        padding_side: Literal["left", "right"] = "right",
        weight_precision_load: torch.dtype = torch.bfloat16,
    ) -> None:
        self.pad_token = pad_token
        self.padding_side = padding_side
        self.source = source
        self.save_path = save_path
        self.bnb_config = bnb_config

        if self.bnb_config is not None:
            logger.info(
                "LlaMA will be quantised following the given configuration."
            )

        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            with_casual_lm=True,
            quantization_config=self.bnb_config,
        )

        if self.bnb_config is None:
            self.model = self.model.to(weight_precision_load)
            logger.info(
                f"Converted {self.__class__.__name__} to {weight_precision_load}"
            )

        self.load_tokenizer(source=source, pad_token=self.pad_token)
        self.model.resize_token_embeddings(
            len(self.tokenizer), pad_to_multiple_of=8
        )  # TODO: optimal value of pad_to_multiple_of varies? see https://github.com/huggingface/tokenizers/issues/991.

    def forward(self, **kwargs):
        """This function wraps the HuggingFace forward function. See the HuggingFace documentation of your Llama model of interest to know which
        parameters to pass, typically the input tokens or embeddings and attention masks.

        Arguments
        ---------
        **kwargs : list
            Please refer to HuggingFace documentation and map it to your Llama model of interest.

        Returns
        -------
        output : torch.Tensor
            This depends on the Llama model. Please refer to the HuggingFace documentation.
        """

        return self.model(**kwargs)

    def generate(self, **kwargs):
        """This function wraps the HuggingFace generate function. See the HuggingFace documentation of your Llama model of interest to know which
        parameters to pass, typically the input tokens or embeddings, attention masks and a transformers.GenerationConfig.

        Arguments
        ---------
        **kwargs : list
            Please refer to HuggingFace documentation and map it to your Llama model of interest.

        Returns
        -------
        hyp : torch.Tensor
            Contains tokenized (indices) outputs.
        """

        with torch.no_grad():
            return self.model.generate(**kwargs)
