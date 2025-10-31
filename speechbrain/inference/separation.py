"""Specifies the inference interfaces for speech separation modules.

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022, 2023
 * Pooneh Mousavi 2023
 * Sylvain de Langen 2023
 * Adel Moumen 2023
 * Pradnya Kandarkar 2023
"""

import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.fetching import LocalStrategy, fetch


class SepformerSeparation(Pretrained):
    """A "ready-to-use" speech separation model.

    Uses Sepformer architecture.

    Example
    -------
    >>> tmpdir = getfixture("tmpdir")
    >>> model = SepformerSeparation.from_hparams(
    ...     source="speechbrain/sepformer-wsj02mix", savedir=tmpdir
    ... )
    >>> mix = torch.randn(1, 400)
    >>> est_sources = model.separate_batch(mix)
    >>> print(est_sources.shape)
    torch.Size([1, 400, 2])
    """

    MODULES_NEEDED = ["encoder", "masknet", "decoder"]

    def separate_batch(self, mix):
        """Run source separation on batch of audio.

        Arguments
        ---------
        mix : torch.Tensor
            The mixture of sources.

        Returns
        -------
        tensor
            Separated sources
        """

        # Separation
        mix = mix.to(self.device)
        mix_w = self.mods.encoder(mix)
        est_mask = self.mods.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.mods.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        return est_source

    def separate_file(self, path, savedir=None):
        """Separate sources from file.

        Arguments
        ---------
        path : str
            Path to file which has a mixture of sources. It can be a local
            path, a web url, or a huggingface repo.
        savedir : path
            Path where to store the wav signals (when downloaded from the web).
        Returns
        -------
        tensor
            Separated sources
        """
        source, fl = split_path(path)
        path = fetch(
            fl,
            source=source,
            savedir=savedir,
            local_strategy=LocalStrategy.SYMLINK,
        )

        batch, fs_file = torchaudio.load(path)
        batch = batch.to(self.device)
        fs_model = self.hparams.sample_rate

        # resample the data if needed
        if fs_file != fs_model:
            print(f"Resampling the audio from {fs_file} Hz to {fs_model} Hz")
            tf = torchaudio.transforms.Resample(
                orig_freq=fs_file, new_freq=fs_model
            ).to(self.device)
            batch = batch.mean(dim=0, keepdim=True)
            batch = tf(batch)

        est_sources = self.separate_batch(batch)
        est_sources = (
            est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
        )
        return est_sources

    def forward(self, mix):
        """Runs separation on the input mix"""
        return self.separate_batch(mix)
