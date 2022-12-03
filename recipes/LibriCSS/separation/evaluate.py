#!/usr/bin/env python
"""
Authors
    Martin Kocour
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import logging
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataset import DynamicItemDataset

import wandb


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, stage):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
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

    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        raise NotImplementedError("Training is not supported now")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        if not isinstance(batch, PaddedBatch):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].squeeze(0)
            batch = PaddedBatch([batch])

        snt_id = batch.id
        mixture = batch.mix_sig

        with torch.no_grad():
            predictions = self.compute_forward(mixture, stage)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, predictions)

        return torch.tensor(torch.nan)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TEST:
            if hasattr(self, "wandb_table") and self.wandb_table is not None:
                wandb.log(
                    {"test_samples": self.wandb_table}, step=epoch, commit=True
                )
                self.wandb_table = None

    def save_audio(self, snt_id, mixture, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if not hasattr(self, "wandb_table") or self.wandb_table is None:
            columns = [f"est_source{i}" for i in range(self.hparams.num_spks)]
            columns = ["id"] + columns + ["mixture"]
            self.wandb_table = wandb.Table(columns=columns)

        data = [snt_id]
        for ns in range(self.hparams.num_spks):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )
            data.append(
                wandb.Audio(
                    signal.detach().cpu().numpy(),
                    sample_rate=self.hparams.sample_rate,
                )
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )
        data.append(
            wandb.Audio(
                signal.detach().cpu().numpy(),
                sample_rate=self.hparams.sample_rate,
            )
        )

        self.wandb_table.add_data(*data)


def dataio_prep(hparams):
    test_data = DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline(mix_wav):
        return sb.dataio.dataio.read_audio(mix_wav)

    test_data.add_dynamic_item(audio_pipeline)
    test_data.set_output_keys(["id", "mix_sig"])
    return test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from recipes.LibriCSS.prepare_data import prepare_libricss

    run_on_main(
        prepare_libricss,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "partitions": ["utterances"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
        },
    )

    # train_data, valid_data, test_data = dataio_prep(hparams)
    test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected(
            device=run_opts["device"]
        )

    wandb.init(
        project="SepFormer",
        entity="mato1102",
        config={},
        resume=True,
        name=hparams["experiment_name"],
    )

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Eval
    separator.evaluate(
        test_data, min_key="si-snr", test_loader_kwargs={"shuffle": True}
    )

    wandb.finish()
