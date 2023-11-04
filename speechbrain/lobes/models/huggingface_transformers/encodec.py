"""This lobe enables the integration of huggingface pretrained encodec.

Repository: https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/encodec
Paper: https://arxiv.org/abs/2210.13438

Authors
 * Artem Ploujnikov 2023
"""

from torch import nn
from speechbrain.dataio.dataio import length_to_mask

try:
    from transformers import EncodecModel
except ImportError:
    MSG = "Please install transformers from HuggingFace to use Encodec\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)


DEFAULT_SAMPLE_RATE = 24000


class Encodec(nn.Module):
    """An wrapper for the HuggingFace encodec model

    Arguments
    ---------
    source : str
        a HuggingFace repository identifier or a path
    save_path : str
        the location where the pretrained model will be saved
    sample_rate : int
        the audio sampling rate
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used
        as part of training another model)"""

    def __init__(
        self, source, save_path=None, sample_rate=None, freeze=True,
    ):
        super().__init__()
        self.source = source
        self.save_path = save_path
        self.freeze = freeze
        self.model = EncodecModel.from_pretrained(source, cache_dir=save_path)
        if not sample_rate:
            sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_rate = sample_rate
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, inputs, lengths):
        """Encodes the input audio as tokens

        Arguments
        ---------
        inputs : torch.Tensor
            a (Batch X Samples) tensor of audio
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            a (Batch X Tokens) tensor of audio tokens
        """
        return self.encode(inputs, lengths)

    def encode(self, inputs, length):
        """Encodes the input audio as tokens

        Arguments
        ---------
        inputs : torch.Tensor
            a (Batch X Samples) tensor of audio
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        tokens : torch.Tensor
            a (Batch X Tokens) tensor of audio tokens
        """
        max_len = inputs.size(1)
        mask = length_to_mask(length * max_len, max_len)
        result = self.model.encode(inputs, mask)
        return result["audio_codes"].squeeze(0).transpose(-1, -2)

    def decode(self, tokens, length):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            a tensor of tokens
        length : torch.Tensor
            a tensor of relative lengths

        Returns
        -------
        audio : torch.Tensor
            the audio
        """
        max_len = tokens.size(1)
        mask = length_to_mask(length * max_len, max_len)
        return self.model.decode(
            tokens.unsqueeze(0).transpose(-1, -2), [None], mask
        )
