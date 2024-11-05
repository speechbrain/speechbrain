"""This lobe enables the integration of huggingface pretrained LaBSE models.
Reference: https://arxiv.org/abs/2007.01852

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Ha Nguyen 2023
"""

import os

import torch
import torch.nn.functional as F

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LaBSE(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained LaBSE models.

    Source paper LaBSE: https://arxiv.org/abs/2007.01852
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed text-based sentence-level embeddings generator or can be finetuned.
    It will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "setu4993/LaBSE"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    output_norm : bool (default: True)
        If True, normalize the output.
    Example
    -------
    >>> inputs = ["La vie est belle"]
    >>> model_hub = "setu4993/smaller-LaBSE"
    >>> save_path = "savedir"
    >>> model = LaBSE(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
        output_norm=True,
    ):
        super().__init__(source=source, save_path=save_path, freeze=freeze)

        self.load_tokenizer(source=source)

        self.output_norm = output_norm

    def forward(self, input_texts):
        """This method implements a forward of the labse model,
        which generates sentence-level embeddings from input text.

        Arguments
        ----------
        input_texts (translation): list
            The list of texts (required).
        """

        # Transform input to the right format of the LaBSE model.
        if self.freeze:
            with torch.no_grad():
                # Tokenize the input text before feeding to LaBSE model.
                input_texts = self.tokenizer(
                    input_texts, return_tensors="pt", padding=True
                )
                # Set the right device for the input.
                for key in input_texts.keys():
                    input_texts[key] = input_texts[key].to(
                        device=self.model.device
                    )
                    input_texts[key].requires_grad = False

                embeddings = self.model(**input_texts).pooler_output

                if self.output_norm:
                    # Output normalizing if needed.
                    embeddings = F.normalize(embeddings, p=2)

                return embeddings

        # Tokenize the input text before feeding to LaBSE model.
        input_texts = self.tokenizer(
            input_texts, return_tensors="pt", padding=True
        )
        # Set the right device for the input.
        for key in input_texts.keys():
            input_texts[key] = input_texts[key].to(device=self.model.device)

        embeddings = self.model(**input_texts).pooler_output

        if self.output_norm:
            # Output normalizing if needed.
            embeddings = F.normalize(embeddings, p=2)

        return embeddings
