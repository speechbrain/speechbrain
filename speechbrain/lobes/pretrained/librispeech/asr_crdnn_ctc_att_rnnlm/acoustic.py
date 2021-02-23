"""
Ready to use models for ASR with librispeech

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
"""

import os
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.data_utils import download_from_huggingface


class ASR(torch.nn.Module):
    """A "ready-to-use" English speech recognizer based on the LibriSpeech seq2seq recipe.
    The class can be used either to run only the encoder (encode()) to extract features
    or to run the entire encoder-decoder-attention model (transcribe()) to transcribe the speech.
    It expects input speech signals sampled at 16 kHz. The system achieves
    a WER=3.09% on LibriSpeech test clean. Despite being quite robust, there is no guarantee
    that this model works well for other tasks.
    This class provides two possible way of using our pretrained ASR nodel:
    1. Downloads from HuggingFace and loads the pretrained models.
    2. Downloads from the web (or copy locally) and loads the pretrained models if
    the different checkpoints aren't stored on HuggingFace. This is particularly
    useful for wrapping your own custom ASR pipeline.

    Arguments
    ---------
    hparams_file : str
        Path where the yaml file with the model definition is stored.
        If it is an url, the yaml file is downloaded. If it's an HuggingFace
        path, it should correspond to the huggingface_model provided.
    huggingface_model: str
        Name of the model stored within HuggingFace.
    save_folder : str
        Path where the lm (yaml + model) will be saved (default 'asr_model')
    freeze_params: bool
        If true, the model is frozen and the gradient is not backpropagated
        through the languange model.

    """

    def __init__(
        self,
        hparams_file="acoustic/BPE1000.yaml",
        huggingface_model="sb/asr-crdnn-librispeech",
        save_folder="model_checkpoints",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        self.save_folder = save_folder
        self.save_yaml_filename = "ASR.yaml"

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

        # putting modules on the right device
        # We need to check if DDP has been initialised
        # in order to give the right device
        if torch.distributed.is_initialized():
            self.device = ":".join(
                [self.hparams["device"].split(":")[0], os.environ["LOCAL_RANK"]]
            )
        else:
            self.device = self.hparams["device"]

        # Creating directory where pre-trained models are stored
        if not os.path.isdir(self.hparams["save_folder"]):
            os.makedirs(self.hparams["save_folder"])

        # putting modules on the right device
        self.mod = torch.nn.ModuleDict(self.hparams["modules"]).to(self.device)

        # Load pretrained modules
        self.load_asr()

        # The tokenizer is the one used by the LM
        self.tokenizer = self.hparams["lm_model"].tokenizer

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.mod.asr_model.eval()
            for p in self.mod.asr_model.parameters():
                p.requires_grad = False
            self.mod.lm_model.eval()
            for p in self.mod.lm_model.parameters():
                p.requires_grad = False

    def encode(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states"""
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        feats = self.mod.compute_features(wavs)
        feats = self.mod.normalize(feats, wav_lens)
        encoder_out = self.mod.asr_encoder(feats)
        return encoder_out

    def transcribe(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words"""
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode(wavs, wav_lens)
            predicted_tokens, scores = self.mod.beam_searcher(
                encoder_out, wav_lens
            )

            predicted_words = [
                self.tokenizer.decode_ids(predicted_tokens[i])
                for i in range(len(predicted_tokens))
            ]

        return predicted_words, predicted_tokens

    def load_asr(self):
        """Loads the AM specified in the yaml file"""

        # Load the acoustic model first
        save_filename = "asr_model.ckpt"
        save_model_path = os.path.join(self.save_folder, save_filename)

        if self.hparams["huggingface"]:
            download_from_huggingface(
                self.hparams["huggingface_model"],
                self.hparams["asr_ckpt_file"],
                self.save_folder,
                save_filename,
            )
        else:
            download_file(self.hparams["asr_ckpt_file"], save_model_path)

        self.mod.asr_model.load_state_dict(
            torch.load(save_model_path, map_location=self.device), strict=True
        )

        # Load the normalizer statistics
        save_filename = "normalizer.ckpt"
        save_model_path = os.path.join(self.save_folder, save_filename)

        if self.hparams["huggingface"]:
            download_from_huggingface(
                self.hparams["huggingface_model"],
                self.hparams["normalize_ckpt_file"],
                self.save_folder,
                save_filename,
            )
        else:
            download_file(self.hparams["normalize_ckpt_file"], save_model_path)

        self.hparams["normalize"]._load(save_model_path, 0, self.device)
