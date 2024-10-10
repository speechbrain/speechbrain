"""This lobe enables the integration of generic huggingface pretrained text
encoders (e.g. BERT).

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Sylvain de Langen 2024
"""

from typing import Optional

import torch

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class TextEncoder(HFTransformersInterface):
    """This lobe enables the integration of a generic HuggingFace text encoder
    (e.g. BERT). Requires the `AutoModel` found from the `source` to have a
    `last_hidden_state` key in the output dict.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "google-bert/bert-base"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    num_layers : int, optional
        When specified, and assuming the passed LM can be truncated that way,
        the encoder for the passed model will be truncated to the specified
        layer (mutating it). This means that the embeddings will be those of the
        Nth layer rather than the last layer. The last layer is not necessarily
        the best for certain tasks.
    **kwargs
        Extra keyword arguments passed to the `from_pretrained` function.
    Example
    -------
    >>> inputs = ["La vie est belle"]
    >>> model_hub = "google-bert/bert-base-multilingual-cased"
    >>> save_path = "savedir"
    >>> model = TextEncoder(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            source=source, save_path=save_path, freeze=freeze, **kwargs
        )

        self.load_tokenizer(source=source)

        if num_layers is not None:
            self.truncate(num_layers)

    def truncate(self, keep_layers: int):
        """Truncates the encoder to a specific layer so that output embeddings
        are the hidden state of the n-th layer.

        Arguments
        ---------
        keep_layers : int
            Number of layers to keep, e.g. 4 would keep layers `[0, 1, 2, 3]`.
        """

        assert (
            keep_layers > 0
        ), "Invalid requested layer count: Must keep at least one LM layer (negative values are not allowed)"
        assert keep_layers <= len(
            self.model.encoder.layer
        ), "Too few layers in LM: kept layer count requested is too high"
        self.model.encoder.layer = self.model.encoder.layer[:keep_layers]

    def forward(self, input_texts, return_tokens: bool = False):
        """This method implements a forward of the encoder model,
        which generates batches of embeddings embeddings from input text.

        Arguments
        ---------
        input_texts : list of str
            The list of texts (required).
        return_tokens : bool
            Whether to also return the tokens.

        Returns
        -------
        (any, torch.Tensor) if `return_tokens == True`
            Respectively:
            - Tokenized sentence in the form of a padded batch tensor. In the HF
              format, as returned by the tokenizer.
            - Output embeddings of the model (i.e. the last hidden state)

        torch.Tensor if `return_tokens` == False
            Output embeddings of the model (i.e. the last hidden state)
        """

        with torch.set_grad_enabled(not self.freeze):
            input_texts = self.tokenizer(
                input_texts, return_tensors="pt", padding=True
            ).to(self.model.device)

            embeddings = self.model(**input_texts).last_hidden_state

            if return_tokens:
                return input_texts, embeddings

            return embeddings
