#!/usr/bin/env/python3
"""This minimal example trains an autoencoder over speech features. The encoder
is a MLP that transforms the input into a lower-dimensional latent representation.
The decoder is another MLP that predicts the input features. The system is trained
with MSE. Given the tiny dataset, the expected behavior is to overfit the
training data  (with a validation performance that stays high).
"""

import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class AutoBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Appling encoder and decoder to the input features"
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        encoded = self.modules.linear1(feats)
        encoded = self.hparams.activation(encoded)
        decoded = self.modules.linear2(encoded)

        return decoded

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the MSE loss."
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        self.mse_metric.append(batch.id, predictions, feats, lens)
        return self.hparams.compute_cost(predictions, feats, lens)

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        self.mse_metric = self.hparams.loss_tracker()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if self.hparams.use_tensorboard:
            if stage == sb.Stage.TRAIN:
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch},
                    train_stats={"loss": self.mse_metric.scores},
                )
            elif stage == sb.Stage.VALID:
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch},
                    valid_stats={"loss": self.mse_metric.scores},
                )
            if stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    {}, test_stats={"loss": self.mse_metric.scores}
                )

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Completed epoch %d" % epoch)
            print("Train loss: %.3f" % self.train_loss)
            print("Valid loss: %.3f" % stage_loss)
        if stage == sb.Stage.TEST:
            print("Test loss: %.3f" % stage_loss)


def data_prep(data_folder):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "train.json",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "dev.json",
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    return train_data, valid_data


def main():
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder)

    # Trainer initialization
    auto_brain = AutoBrain(hparams["modules"], hparams["opt_class"], hparams)

    # Training/validation loop
    auto_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    auto_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    assert auto_brain.train_loss < 0.08


if __name__ == "__main__":
    main()


def test_error():
    main()
