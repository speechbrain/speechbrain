"""
Pre-trained LM on LibriSpeech for inference.

Example
-------
>>> import torch
>>> from pretrained import RNNLM
>>> lm = RNNLM()

>>> # Next word prediction
>>> text = "THE CAT IS ON"
>>> encoded_text = lm.tokenizer.encode_as_ids(text)
>>> encoded_text = torch.Tensor(encoded_text).unsqueeze(0)
>>> prob_out, _ = lm(encoded_text.to(lm.device))
>>> index = int(torch.argmax(prob_out[0,-1,:]))
>>> print(lm.tokenizer.decode(index))

>>> # Text Generation
>>> encoded_text = torch.tensor([0, 2]) # bos token + the
>>> encoded_text = encoded_text.unsqueeze(0).to(lm.device)
>>>
>>> for i in range(19):
>>>     prob_out, _ = lm(encoded_text)
>>>     index = torch.argmax(prob_out[0,-1,:]).unsqueeze(0)
>>>     encoded_text = torch.cat([encoded_text, index.unsqueeze(0)], dim=1)
>>>
>>> encoded_text = encoded_text[0,1:].tolist()
>>> print(lm.tokenizer.decode(encoded_text))


Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import os
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
import sentencepiece as spm


class RNNLM(torch.nn.Module):
    def __init__(
        self,
        hparams_file="hparams/pretrained_RNNLM.yaml",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            self.hparams = sb.load_extended_yaml(fin, overrides)

        self.device = self.hparams["device"]

        # Creating directory where pre-trained models are stored
        if not os.path.isabs(self.hparams["save_folder"]):
            dirname = os.path.dirname(__file__)
            self.hparams["save_folder"] = os.path.join(
                dirname, self.hparams["save_folder"]
            )
        if not os.path.isdir(self.hparams["save_folder"]):
            os.makedirs(self.hparams["save_folder"])

        # putting modules on the right device
        self.net = self.hparams["net"].to(self.device)

        # Load pretrained modules
        self.load_lm()

        # Load tokenizer
        self.load_tokenizer()

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, x, hx=None):
        """Compute the LM probabilities given and encoded input."""
        return self.net.forward(x, hx)

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(self.hparams["save_folder"], "lm.ckpt")
        download_file(self.hparams["lm_ckpt_file"], save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.net.load_state_dict(state_dict, strict=True)

    def load_tokenizer(self):
        """Loads the sentence piece tokenizer specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams["save_folder"], "tokenizer.model",
        )

        # Downloading from the web
        download_file(
            source=self.hparams["tok_mdl_file"], dest=save_model_path,
        )

        # Initialize and pre-train the tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(save_model_path)
