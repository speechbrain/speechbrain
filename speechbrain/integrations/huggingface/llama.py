"""This lobe enables the integration of huggingface pretrained LlaMA models.

Authors
 * Titouan Parcollet 2025
 * Shucong Zhang 2025
 * Pooneh Mousavi 2023
 * Adel Moumen 2025
"""

from typing import List

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
    pad_token : str (default: "[PAD]")
        String representation of the padding token. This may change from one model to another.
    torch_dtype : torch.dtype (default: torch.float16)
        If no bnb_config is given, this parameter defines the loading type of the parameters of the model. This is useful to reduce memory footprint, but it does not change the compute dtype. For this just refer to mixed precision training in SpeechBrain.
    additional_special_tokens : List[str], optional
        A list of additional special tokens to add to the tokenizer. These tokens will be added using the tokenizer's `add_special_tokens` method.
    pad_to_multiple_of : int (default: 8)
        The token embeddings will be resized to a multiple of this value. This is useful to maximise the use of tensor cores on modern GPUs.
    **kwargs : dict
        Extra keyword arguments passed to the `from_pretrained` function. This can be used, for instance, to change the type of attention. The HuggingFace documentation gives the full dict of parameters which may be model dependent.

    Example
    -------
    >>> model_hub = "meta-llama/Llama-2-7b-chat-hf"
    >>> save_path = "savedir"
    >>> model = LLaMA(model_hub, save_path)  # doctest: +SKIP
    >>> tokens = torch.tensor([[1, 1]])
    >>> attention_mask = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, attention_mask)  # doctest: +SKIP
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        bnb_config: BitsAndBytesConfig = None,
        freeze: bool = False,
        pad_token: str = "[PAD]",
        torch_dtype: torch.dtype = torch.float16,
        additional_special_tokens: List[str] = None,
        pad_to_multiple_of: int = 8,
        **kwargs,
    ) -> None:
        self.pad_token = pad_token
        self.source = source
        self.save_path = save_path
        self.bnb_config = bnb_config

        # Capture config-only overrides to avoid passing them to from_pretrained
        self._config_overrides = {}
        if "output_hidden_states" in kwargs:
            self._config_overrides["output_hidden_states"] = kwargs.pop(
                "output_hidden_states"
            )

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
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.load_tokenizer(source=source, pad_token=self.pad_token)

        if additional_special_tokens is not None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens}
            )

        # We resize the token embeddings size to a factor of 8 to maximise
        # the use of tensorcores.
        # Note: resize_token_embeddings may require float32 for some operations
        # (e.g., Cholesky decomposition), so we temporarily convert to float32
        # if the model is in bfloat16, then convert back.
        # Skip dtype conversion if model is quantized (bnb_config is set)
        original_dtype = None
        model_needs_conversion = False
        if self.bnb_config is None and torch_dtype == torch.bfloat16:
            # Check if model is actually in bfloat16
            if hasattr(self.model, "get_input_embeddings"):
                embedding_layer = self.model.get_input_embeddings()
                if (
                    embedding_layer is not None
                    and embedding_layer.weight.dtype == torch.bfloat16
                ):
                    model_needs_conversion = True
                    original_dtype = torch.bfloat16
                    # Temporarily convert entire model to float32 for resize operation
                    # This is necessary because resize_token_embeddings performs operations
                    # (like Cholesky decomposition) that require float32
                    self.model = self.model.to(torch.float32)

        self.model.resize_token_embeddings(
            len(self.tokenizer), pad_to_multiple_of=pad_to_multiple_of
        )

        # Convert back to original dtype if we changed it
        if model_needs_conversion and original_dtype == torch.bfloat16:
            self.model = self.model.to(original_dtype)

    def override_config(self, config):
        """Users should modify this function according to their own tasks.

        Arguments
        ---------
        config : HuggingFace config object
            The original config.

        Returns
        -------
        config : HuggingFace config object
            Overridden config.
        """
        # Apply user-specified config overrides captured from kwargs
        for key, value in getattr(self, "_config_overrides", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(
                    f"Config has no attribute '{key}', cannot apply override."
                )
        return config

    def forward(self, **kwargs):
        """This function wraps the HuggingFace forward function. See the HuggingFace documentation of your Llama model of interest to know which
        parameters to pass, typically the input tokens or embeddings and attention masks.

        Arguments
        ---------
        **kwargs : dict
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
        **kwargs : dict
            Please refer to HuggingFace documentation and map it to your Llama model of interest.

        Returns
        -------
        hyp : torch.Tensor
            Contains tokenized (indices) outputs.
        """

        with torch.no_grad():
            return self.model.generate(**kwargs)
