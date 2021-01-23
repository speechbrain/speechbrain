"""
Pre-trained LM on LibriSpeech for inference.

Example
-------
>>> import torch
>>> from pretrained import LM
>>> # RNN LM
>>> lm = LM('hparams/pretrained_RNNLM_BPE1000.yaml')
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
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml


class LM(torch.nn.Module):
    def __init__(
        self,
        hparams_file="hparams/pretrained_RNNLM_BPE1000.yaml",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        # Loading modules defined in the yaml file
        if not os.path.isabs(hparams_file):
            dirname = os.path.dirname(__file__)
            hparams_file = os.path.join(dirname, hparams_file)
        with open(hparams_file) as fin:
            self.hparams = load_hyperpyyaml(fin, overrides)

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
        self.model = self.hparams["model"].to(self.device)

        # Load pretrained modules
        self.load_lm()

        # Load tokenizer
        self.tokenizer = self.hparams["tokenizer"].spm

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, hx=None):
        """Compute the LM probabilities given and encoded input."""
        return self.model.forward(x, hx)

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(self.hparams["save_folder"], "lm.ckpt")
        download_file(self.hparams["lm_ckpt_file"], save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
