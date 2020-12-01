"""
A "ready-to-use" speech recognizer based on seq2seq a trained with librispeech.
The class expects input speech signals sampled at 16 kHz. The system achieves
a WER=3.07% on LibriSpeech test clean. Despite being quite robust, there is no warranty
that this model works well for other tasks.

Example
-------
>>> import torchaudio
>>> do_ASR = ASR_infer()
>>> audio_file='../../../../samples/audio_samples/example2.flac'
>>> wav, fs = torchaudio.load(audio_file)
>>> wav_lens = torch.tensor([1]).float()
>>> words, tokens = do_ASR(wav, wav_lens)
>>> words
[['MY', 'FATHER', 'HAS', 'REVEALED', 'THE', "CULPRIT'S", 'NAME']]


Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import os
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.tokenizers.SentencePiece import SentencePiece


class ASR_infer(torch.nn.Module):
    def __init__(self, hparams_file="hparams/infer.yaml"):
        """Download and Pretraining of the moduels specified in the yaml"""
        super().__init__()

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            self.hparams = sb.load_extended_yaml(fin)

        self.device = self.hparams["device"]

        # Creating directory where pre-trained models are stored
        if not os.path.isdir(self.hparams["save_folder"]):
            os.makedirs(self.hparams["save_folder"])

        # putting modules on the right device
        self.mod = torch.nn.ModuleDict(self.hparams["modules"]).to(self.device)

        # Pretraining modules
        self.load_tokenizer()
        self.load_lm()
        self.load_asr()

    def encode(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states"""
        with torch.no_grad():
            wavs = wavs.float()
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            feats = self.mod.compute_features(wavs)
            feats = self.mod.normalize(feats, wav_lens)
            encoder_out = self.mod.asr_encoder(feats)
        return encoder_out

    def forward(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words"""
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode(wavs, wav_lens)
            predicted_tokens, scores = self.mod.beam_searcher(
                encoder_out, wav_lens
            )

            predicted_words = self.mod.tokenizer(
                predicted_tokens, task="decode_from_list"
            )

        return predicted_words, predicted_tokens

    def load_tokenizer(self):
        """Loads the sentence piece tokinizer specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams["save_folder"],
            str(self.hparams["output_neurons"]) + "_unigram.model",
        )

        # Donwloanding from the web
        download_file(
            source=self.hparams["tok_mdl_file"],
            dest=save_model_path,
            replace_existing=True,
        )

        # Initialize and pre-train the tokenizer
        self.mod.tokenizer = SentencePiece(
            model_dir=self.hparams["save_folder"],
            vocab_size=self.hparams["output_neurons"],
        )
        self.mod.tokenizer.sp.load(save_model_path)

    def load_asr(self):
        """Loads the AM specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams["save_folder"], "asr_model.ckpt"
        )
        if "http" in self.hparams["asr_ckpt_file"]:
            download_file(self.hparams["asr_ckpt_file"], save_model_path)

        self.mod.asr_model.load_state_dict(
            torch.load(save_model_path), strict=True
        )
        self.mod.asr_model.eval()
        for p in self.mod.asr_model.parameters():
            p.requires_grad = False

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams["save_folder"], "lm_model.ckpt"
        )
        download_file(self.hparams["lm_ckpt_file"], save_model_path)

        # Load downloaded model, removing prefix
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.mod.lm_model.load_state_dict(state_dict, strict=True)
        self.mod.lm_model.eval()
        for p in self.mod.lm_model.parameters():
            p.requires_grad = False
