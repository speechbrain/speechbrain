"""
Pre-trained models for speaker verification.

Example
-------
>>> import torch
>>> from speechbrain.dataio.dataio import read_audio
>>> from pretrained import Verification
>>> verification = Verification() # Ecapa model downloaded
>>> # Compute embeddings
>>> signal1 =read_audio('../../../../samples/audio_samples/example1.wav')
>>> signal1 = signal1.unsqueeze(0) # [batch, time]
>>> lens1 = torch.Tensor([1.0])
>>> emb1 = verification.compute_embeddings(signal1, lens1)
>>> signal2 =read_audio('../../../../samples/audio_samples/example2.flac')
>>> signal2 = signal2.unsqueeze(0) # [batch, time]
>>> lens2 = torch.Tensor([1.0])
>>> emb2 = verification.compute_embeddings(signal2, lens2)
>>> # Speaker Verification
>>> score, decision = verification.verify(signal1, lens1, signal2, lens2, 0.5)
>>> print(score)
>>> print(decision)
>>> score, decision = verification.verify(signal1, lens1, signal1, lens1, 0.5)
>>> print(score)
>>> print(decision)
>>> signal_noise = signal1 + torch.rand_like(signal1) * 0.02
>>> score, decision = verification.verify(signal1, lens1, signal_noise, lens1, 0.5)
>>> print(score)
>>> print(decision)


Authors
 * Mirco Ravanelli 2020
"""

import os
import torch
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml


class Verification(torch.nn.Module):
    """Pretrain and use verification systems.

    Arguments
    ---------
    hparams_file : str
        hyper-parameter file that defines the verification system.
    overrides : Dict
        Dictionary with the parameters to override.
    freeze_params : bool
        If true, the parameters are frozen.
    norm_emb: bool
	If True, the embeddings are normalized.
    save_folder : str
        Path where the lm (yaml + model) will be saved (default 'asr_model')
    """
    def __init__(
        self,
        hparams_file="https://www.dropbox.com/s/ct72as3hapy8kb5/ecapa_big.yaml?dl=1",
        overrides={},
        freeze_params=True,
        norm_emb=True,
        save_folder='emb_model'
    ):
        """Downloads the pretrained modules specified in the yaml"""
        super().__init__()
        self.norm_emb = norm_emb

        save_model_path = os.path.join(save_folder, "embedding.yaml")
        download_file(hparams_file, save_model_path)
        hparams_file = save_model_path

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            overrides["save_folder"] = save_folder
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
        self.embedding_model = self.hparams["embedding_model"].to(self.device)
        self.mean_var_norm = self.hparams["mean_var_norm"].to(self.device)
        self.mean_var_norm_emb = self.hparams["mean_var_norm_emb"].to(self.device)
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        # Load pretrained modules
        self.load_model()

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.embedding_model.eval()
            for p in self.embedding_model.parameters():
                p.requires_grad = False

    def compute_embeddings(self, wavs, wav_lens):
        """Compute speaker embeddings.

        Arguments
        ---------
        wavs : Torch.Tensor
            Tensor containing the speech waveform (batch, time).
            Make sure the sample rate is fs=16000 Hz.
        wav_lens: Torch.Tensor
            Tensor containing the relative length for each sentence
            in the length (e.g., [0.8 0.6 1.0])
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        feats = self.hparams["compute_features"](wavs)
        feats = self.hparams["mean_var_norm"](feats, wav_lens)
        embeddings = self.embedding_model(feats, wav_lens)
        if self.norm_emb:
            embeddings = self.hparams["mean_var_norm_emb"](
                embeddings, torch.ones(embeddings.shape[0], device=self.device))
        return embeddings.squeeze(1)

    def verify(self, wavs1, wav1_lens, wavs2, wav2_lens, threshold):
        """Performs speaker verification with cosine distance.
        It returns the score and the decision (0 different speakers,
        1 same speakers).

        Arguments
        ---------
        wavs1 : Torch.Tensor
                Tensor containing the speech waveform1 (batch, time).
                Make sure the sample rate is fs=16000 Hz.
        wav1_lens: Torch.Tensor
                Tensor containing the relative length for each sentence
                in the length (e.g., [0.8 0.6 1.0])
        wavs2 : Torch.Tensor
                Tensor containing the speech waveform2 (batch, time).
                Make sure the sample rate is fs=16000 Hz.
        wav2_lens: Torch.Tensor
                Tensor containing the relative length for each sentence
                in the length (e.g., [0.8 0.6 1.0])
        threshold: Float
                Threshold applied to the cosine distance to decide if the
                speaker is different (0) or the same (1).
        """
        emb1 = self.compute_embeddings(wavs1, wav1_lens)
        emb2 = self.compute_embeddings(wavs2, wav2_lens)
        score = self.similarity(emb1, emb2)
        return score, score > threshold

    def load_model(self):
        """Loads the models specified in the yaml file"""
        # Embedding Model
        save_model_path = os.path.join(
            self.hparams["save_folder"], "embedding_model.ckpt")
        download_file(self.hparams["embedding_model_file"], save_model_path)
        state_dict = torch.load(save_model_path, map_location=self.device)
        self.embedding_model.load_state_dict(state_dict, strict=True)
        
        # Normalization
        if self.norm_emb:
            save_model_path = os.path.join(
                self.hparams["save_folder"], "mean_var_norm_emb.ckpt")
            download_file(self.hparams["embedding_norm_file"], save_model_path)
            self.mean_var_norm_emb._load(save_model_path, 0, self.device)
