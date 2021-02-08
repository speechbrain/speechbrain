"""
Pre-trained LM on LibriSpeech for inference.

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import os
import torch
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml


class LM(torch.nn.Module):
    """Downloads and loads the pretrained language model (LM).

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

    Example:
    ---------
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
    """

    def __init__(
        self,
        hparams_file="https://www.dropbox.com/s/j51op5l7i356s3v/pretrained_RNNLM_BPE1000.yaml?dl=1",
        save_folder="lm_model",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()
        self.save_folder = save_folder

        # Download yaml file from the web
        save_file = os.path.join(save_folder, "LM.yaml")
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
        save_model_path = os.path.join(self.save_folder, "lm.ckpt")
        download_file(self.hparams["lm_ckpt_file"], save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
