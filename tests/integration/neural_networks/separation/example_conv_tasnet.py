#!/usr/bin/env/python3
"""This minimal example trains a speech separation system with on a tiny dataset.
The architecture is based on ConvTasnet and expects in input mixtures of two
speakers.
"""

import torch
import pathlib
import speechbrain as sb
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper


class SepBrain(sb.Brain):
    def compute_forward(self, mixture, stage):
        "Given an input batch it computes the two estimated sources."
        mix_w = self.hparams.encoder(mixture)
        est_mask = self.hparams.mask_net(mix_w)
        mix_w = torch.stack([mix_w] * 2)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [self.hparams.decoder(sep_h[i]).unsqueeze(-1) for i in range(2)],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_conv = est_source.size(1)
        if T_origin > T_conv:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_conv))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

    def compute_objectives(self, predictions, targets):
        "Given the network predictions and targets computed the PIT loss."
        loss = get_si_snr_with_pitwrapper(targets, predictions)
        return loss

    def fit_batch(self, batch):
        """Fits a training batch."""
        inputs = batch.mix_sig.data.to(self.device)
        targets = torch.cat(
            [
                batch.source1.data.unsqueeze(-1),
                batch.source2.data.unsqueeze(-1),
            ],
            dim=-1,
        ).to(self.device)

        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Evaluates a batch"""
        inputs = batch.mix_sig.data.to(self.device)
        targets = torch.cat(
            [
                batch.source1.data.unsqueeze(-1),
                batch.source2.data.unsqueeze(-1),
            ],
            dim=-1,
        ).to(self.device)

        predictions = self.compute_forward(inputs, stage)
        loss = self.compute_objectives(predictions, targets)
        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            print("Completed epoch %d" % epoch)
            print("Train SI-SNR: %.3f" % -self.train_loss)
            print("Valid SI-SNR: %.3f" % -stage_loss)
        elif stage == sb.Stage.TEST:
            print("Test SI-SNR: %.3f" % -stage_loss)


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=data_folder / "minimal_example_convtasnet_tr.csv",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=data_folder / "minimal_example_convtasnet_cv.csv",
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("mix_wav", "s1_wav", "s2_wav")
    @sb.utils.data_pipeline.provides("mix_sig", "source1", "source2")
    def audio_pipeline(mix_wav, s1_wav, s2_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        yield mix_sig
        source1 = sb.dataio.dataio.read_audio(s1_wav)
        yield source1
        source2 = sb.dataio.dataio.read_audio(s2_wav)
        yield source2

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "mix_sig", "source1", "source2"]
    )

    return train_data, valid_data


def main():
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = "../../../../samples/audio_samples/sourcesep_samples"
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    sep_brain = SepBrain(hparams["modules"], hparams["opt_class"], hparams)

    # Training/validation loop
    sep_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    sep_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    assert sep_brain.train_loss < 5.0


if __name__ == "__main__":
    main()


def test_error():
    main()
