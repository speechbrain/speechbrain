"""
LM trained on Timers and Such.

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import os
import torch
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml


class LM(torch.nn.Module):
    """Downloads and loads the pretrained language model (LM) for
       the timers-and-such dataset.

    Arguments
    ---------
    hparams_file : str
        Path where the yaml file with the model definition is stored.
        If it is an url, the yaml file is downloaded.
    save_folder : str
        Path where the lm (yaml + model) will be saved (default 'lm_model')
    freeze_params: bool
        If true, the model is frozen and the gradient is not backpropagated
        through the languange model.

    Example
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

    """

    def __init__(
        self,
        hparams_file="hparams/pretrained.yaml",
        save_folder="lm_TAS",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        self.save_folder = save_folder

        # Download yaml file from the web
        save_file = os.path.join(save_folder, "LM_TAS.yaml")
        download_file(hparams_file, save_file)
        hparams_file = save_file

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            overrides["save_folder"] = save_folder
            self.hparams = load_hyperpyyaml(fin, overrides)

        if not os.path.isdir(self.hparams["save_folder"]):
            os.makedirs(self.hparams["save_folder"])

        # putting modules on the right device
        self.device = self.hparams["device"]
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
