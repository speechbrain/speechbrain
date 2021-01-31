"""
Ready to use models for ASR with librispeech

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import os
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file


class ASR(torch.nn.Module):
    """A "ready-to-use" English speech recognizer based on the LibriSpeech seq2seq recipe.
    The class can be used either to run only the encoder (encode()) to extract features
    or to run the entire encoder-decoder-attention model (transcribe()) to transcribe the speech.
    It expects input speech signals sampled at 16 kHz. The system achieves
    a WER=3.09% on LibriSpeech test clean. Despite being quite robust, there is no guarantee
    that this model works well for other tasks.

    Arguments
    ---------
    hparams_file : str
        Path where the yaml file with the model definition is stored.
        If it is an url, the yaml file is downloaded.
    save_folder : str
        Path where the lm (yaml + model) will be saved (default 'asr_model')
    freeze_params: bool
        If true, the model is frozen and the gradient is not backpropagated
        through the languange model.

    >>> import torch
    >>> import torchaudio
    >>> from pretrained import ASR
    >>> asr_model = ASR()
    >>> audio_file='../../../../../samples/audio_samples/example2.flac'
    >>> # Make sure your output is sampled at 16 kHz
    >>> wav, fs = torchaudio.load(audio_file)
    >>> wav_lens = torch.tensor([1]).float()
    >>> words, tokens = asr_model.transcribe(wav, wav_lens)
    >>> words
    [['MY', 'FATHER', 'HAS', 'REVEALED', 'THE', "CULPRIT'S", 'NAME']]
    """

    def __init__(
        self,
        hparams_file="https://www.dropbox.com/s/54vmm04g3gezwz3/pretrained_ASR_BPE1000.yaml?dl=1",
        save_folder="asr_model",
        overrides={},
        freeze_params=True,
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()

        save_model_path = os.path.join(save_folder, "ASR.yaml")
        download_file(hparams_file, save_model_path)
        hparams_file = save_model_path

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            overrides["save_folder"] = save_folder
            self.hparams = load_hyperpyyaml(fin, overrides)

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
        save_model_path = os.path.join(
            self.hparams["save_folder"], "asr_model.ckpt"
        )
        download_file(self.hparams["asr_ckpt_file"], save_model_path)

        self.mod.asr_model.load_state_dict(
            torch.load(save_model_path, map_location=self.device), strict=True
        )

        save_model_path = os.path.join(
            self.hparams["save_folder"], "normalizer.ckpt"
        )
        download_file(self.hparams["normalize_file"], save_model_path)
        self.hparams["normalize"]._load(save_model_path, 0, self.device)
