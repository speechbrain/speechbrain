"""
Pre-trained LM on LibriSpeech for inference.

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
"""

import os
import torch
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.data_utils import download_from_huggingface
from hyperpyyaml import load_hyperpyyaml


class LM(torch.nn.Module):
    """ This class provides two possible way of using pretrained LM:
    1. Downloads from HuggingFace and loads the pretrained language model (LM).
    2. Downloads from the web (or copy locally) and loads a pretrained language
    model if the checkpoint isn't stored on HuggingFace. This is particularly
    useful for LM not ditributed by SpeechBrain.

    Arguments
    ---------
    hparams_file : str
        Path where the yaml file with the model definition is stored.
        If it is an url, the yaml file is downloaded. If it's an HuggingFace
        path, it should correspond to the huggingface_model provided.
    huggingface_model: str
        Name of the model stored within HuggingFace.
    save_folder : str
        Path where the lm (yaml + model) will be saved
        (default 'model_checkpoints')
    freeze_params: bool
        If true, the model is frozen and the gradient is not backpropagated
        through the languange model.

    Example:
    ---------
    >>> import torch
    >>> from pretrained import LM
    >>> # RNN LM
    >>> lm = LM()
    >>> # Next word prediction
    >>> text = "THE CAT IS ON"
    >>> encoded_text = lm.tokenizer.encode_as_ids(text)
    >>> encoded_text = torch.Tensor(encoded_text).unsqueeze(0)
    >>> prob_out, _ = lm(encoded_text.to(lm.device))
    >>> index = int(torch.argmax(prob_out[0,-1,:]))
    >>> print(lm.tokenizer.decode(index))
    THE
    >>> # Text Generation
    >>> encoded_text = torch.tensor([0, 2]) # bos token + the
    >>> encoded_text = encoded_text.unsqueeze(0).to(lm.device)
    >>> for i in range(19):
    >>>     prob_out, _ = lm(encoded_text)
    >>>     index = torch.argmax(prob_out[0,-1,:]).unsqueeze(0)
    >>>     encoded_text = torch.cat([encoded_text, index.unsqueeze(0)], dim=1)
    >>>
    >>> encoded_text = encoded_text[0,1:].tolist()
    >>> print(lm.tokenizer.decode(encoded_text))
    THE NEXT DAY THE SKY WAS CLEAR AND THE SUN WAS SHINING BRIGHTLY
    """

    def __init__(
        self,
        hparams_file="lm/RNNLM_BPE1000.yaml",
        huggingface_model="sb/asr-crdnn-librispeech",
        save_folder="model_checkpoints",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        self.save_folder = save_folder
        self.save_yaml_filename = "LM.yaml"

        # Download yaml file from huggingface or elsewhere
        save_file = os.path.join(save_folder, self.save_yaml_filename)
        if huggingface_model is not None:
            download_from_huggingface(
                huggingface_model,
                hparams_file,
                self.save_folder,
                self.save_yaml_filename,
            )
        else:
            download_file(hparams_file, save_file)

        hparams_file = save_file

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            overrides["save_folder"] = save_folder
            self.hparams = load_hyperpyyaml(fin, overrides)

        if not os.path.isdir(self.hparams["save_folder"]):
            os.makedirs(self.hparams["save_folder"])

        # putting modules on the right device
        # We need to check if DDP has been initialised
        # in order to give the right device
        if torch.distributed.is_initialized():
            self.device = ":".join(
                [self.hparams["device"].split(":")[0], os.environ["LOCAL_RANK"]]
            )
        else:
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
        """ Downloads the LM given in the YAML file"""

        save_filename = "lm.ckpt"
        save_model_path = os.path.join(self.save_folder, save_filename)

        if self.hparams["huggingface"]:
            download_from_huggingface(
                self.hparams["huggingface_model"],
                self.hparams["lm_ckpt_file"],
                self.save_folder,
                save_filename,
            )
        else:
            download_file(self.hparams["lm_ckpt_file"], save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
