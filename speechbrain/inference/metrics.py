"""Specifies the inference interfaces for metric estimation modules.

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

from speechbrain.inference.interfaces import Pretrained


class SNREstimator(Pretrained):
    """A "ready-to-use" SNR estimator."""

    MODULES_NEEDED = ["encoder", "encoder_out"]
    HPARAMS_NEEDED = ["stat_pooling", "snrmax", "snrmin"]

    def estimate_batch(self, mix, predictions):
        """Run SI-SNR estimation on the estimated sources, and mixture.

        Arguments
        ---------
        mix : torch.Tensor
            The mixture of sources of shape B X T
        predictions : torch.Tensor
            of size (B x T x C),
            where B is batch size
                  T is number of time points
                  C is number of sources

        Returns
        -------
        tensor
            Estimate of SNR
        """

        predictions = predictions.permute(0, 2, 1)
        predictions = predictions.reshape(-1, predictions.size(-1))

        if hasattr(self.hparams, "separation_norm_type"):
            if self.hparams.separation_norm_type == "max":
                predictions = (
                    predictions / predictions.max(dim=1, keepdim=True)[0]
                )
                mix = mix / mix.max(dim=1, keepdim=True)[0]

            elif self.hparams.separation_norm_type == "stnorm":
                predictions = (
                    predictions - predictions.mean(dim=1, keepdim=True)
                ) / predictions.std(dim=1, keepdim=True)
                mix = (mix - mix.mean(dim=1, keepdim=True)) / mix.std(
                    dim=1, keepdim=True
                )

        min_T = min(predictions.shape[1], mix.shape[1])
        assert predictions.shape[1] == mix.shape[1], "lengths change"

        mix_repeat = mix.repeat(2, 1)
        inp_cat = torch.cat(
            [
                predictions[:, :min_T].unsqueeze(1),
                mix_repeat[:, :min_T].unsqueeze(1),
            ],
            dim=1,
        )

        enc = self.mods.encoder(inp_cat)
        enc = enc.permute(0, 2, 1)
        enc_stats = self.hparams.stat_pooling(enc)

        # this gets the SI-SNR estimate in the compressed range 0-1
        snrhat = self.mods.encoder_out(enc_stats).squeeze()

        # get the SI-SNR estimate in the true range
        snrhat = self.gettrue_snrrange(snrhat)
        return snrhat

    def forward(self, mix, predictions):
        """Just run the batch estimate"""
        return self.estimate_batch(mix, predictions)

    def gettrue_snrrange(self, inp):
        """Convert from 0-1 range to true snr range"""
        range = self.hparams.snrmax - self.hparams.snrmin
        inp = inp * range
        inp = inp + self.hparams.snrmin
        return inp
