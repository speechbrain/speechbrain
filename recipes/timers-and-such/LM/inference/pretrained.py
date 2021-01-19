"""
LM trained on Timers and Such.

Example usage (text generation):
-------

>>> import torch
>>> from pretrained import LM
>>> lm = LM('hparams/pretrained.yaml')
>>>
>>> text = "SET A"
>>> encoded_text = lm.tokenizer.encode_as_ids(text)
>>> encoded_text = torch.tensor([0] + encoded_text) # bos token + the
>>> encoded_text = encoded_text.unsqueeze(0).to(lm.device)
>>>
>>> for i in range(19):
>>>     prob_out, _ = lm(encoded_text)
>>>     index = torch.argmax(prob_out[0,-1,:]).unsqueeze(0)
>>>     encoded_text = torch.cat([encoded_text, index.unsqueeze(0)], dim=1)
>>>
>>>
>>> encoded_text = encoded_text[0,1:].tolist()
>>> print(lm.tokenizer.decode(encoded_text)) # Should be something like "SET A TIMER FOR ONE HOUR AND SEVEN MINUTES ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇  ⁇ "

Authors
 * Loren Lugosch 2020
"""

import os
import torch
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml


class LM(torch.nn.Module):
    def __init__(
        self,
        hparams_file="hparams/pretrained.yaml",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        # Loading modules defined in the yaml file
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
        self.net = self.hparams["net"].to(self.device)

        # Load pretrained modules
        self.load_lm()

        # Load tokenizer
        self.tokenizer = self.hparams["tokenizer"].spm

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, x, hx=None):
        return self.net.forward(x, hx)

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(self.hparams["save_folder"], "lm.ckpt")
        download_file(self.hparams["lm_ckpt_file"], save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.net.load_state_dict(state_dict, strict=True)
