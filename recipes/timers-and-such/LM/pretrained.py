"""
LM trained on Timers and Such

Example
-------
>>> import torchaudio
>>> lm = LM()

TODO forward()?

Authors
 * Loren Lugosch 2020
"""

import os
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file


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
