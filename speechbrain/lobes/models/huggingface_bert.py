"""This lobe enables the integration of huggingface pretrained wav2vec2 models.
Reference: https://arxiv.org/abs/1810.04805
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html
Authors
 * Abdelmoumene Boumadane 2021
 * AbdelWahab Heba 2021
"""

from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("Please install transformer from HuggingFace to use wav2vec2!")


class HuggingFaceBert(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained Bert based  models.

    Source paper: https://arxiv.org/abs/1810.04805
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html
    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.
    Arguments
    ---------

    source : str
        HuggingFace hub name: e.g "bert-base-cased"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    """

    def __init__(self, source, save_path, freeze=True):

        super(HuggingFaceBert, self).__init__()

        self.bert = AutoModel.from_pretrained(source, cache_dir=save_path)
        self.freeze = freeze

        if self.freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, input_masks):
        """Takes an input a batch of input IDS and mask

        Arguments
        ---------
        input_ids : Tensor
            bert token_ids
        input_masks : Tensor
            input_masks of the input tokens

        """

        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=input_masks
        )

        # The last hidden states of all tokens
        output = pooled_output[0]

        return output


class HuggingFaceBertToknizer:

    """This Tokenizer helps using the BertBase models Tokenizers to be used as inputs to Bert Based models.

    Arguments
    ---------

    source : str
        HuggingFace hub name: e.g "bert-base-cased"
    save_path : str
        Path (dir) of the downloaded model.
    """

    def __init__(self, source, save_path, freeze=True):
        super(HuggingFaceBertToknizer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            source, cache_dir=save_path
        )

    def batch_encode(self, batch):
        """Encoding of a batch of strings to the Bert Expected input

        Arguments
        ---------

        Batch : List(str)
            Batch of strings to be tokenize
        save_path : str
            Path (dir) of the downloaded model.

        Returns
        -------
        (inputs,attention_mask)
            return input token Ids and the attention mask
        """
        encoded_batch = self.tokenizer.batch_encode_plus(batch, padding=True)
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        # Bert based models only process sequences of length 512
        input_ids = [t[: min(len(t), 512)] for t in input_ids]
        attention_mask = [t[: min(len(t), 512)] for t in attention_mask]

        return input_ids, attention_mask

    def encode(self, transcript):
        """Encoding of a batch of strings to the Bert Expected input

        Arguments
        ---------

        Batch : List(str)
            Batch of strings to be tokenize
        save_path : str
            Path (dir) of the downloaded model.

        Returns
        -------
        (inputs,attention_mask)
            return input token Ids and the attention mask
        """
        input_ids = self.tokenizer.encode(transcript)

        # Bert based models only process sequences of length 512
        input_ids = input_ids[: min(len(input_ids), 512)]
        attention_mask = [1] * len(input_ids)

        return input_ids, attention_mask
