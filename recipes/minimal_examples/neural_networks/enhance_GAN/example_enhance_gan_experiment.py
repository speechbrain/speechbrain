#!/usr/bin/env/python3
"""This minimal example trains a GAN speech enhancement system on a tiny dataset.
The generator and the discriminator are based on convolutional networks.
"""

import torch
import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class EnhanceGanBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the enhanced signal"
        wavs, lens = batch.sig

        noisy = self.hparams.add_noise(wavs, lens).unsqueeze(-1)
        enhanced = self.modules.generator(noisy)

        return enhanced

    def compute_objectives(self, predictions, batch, stage, optim_name=""):
        "Given the network predictions and targets computed the total loss"
        clean_wavs, lens = batch.sig
        batch_size = clean_wavs.size(0)

        # Average the predictions of each time step
        clean_wavs = clean_wavs.unsqueeze(-1)
        real_result = self.modules.discriminator(clean_wavs).mean(dim=1)
        simu_result = self.modules.discriminator(predictions).mean(dim=1)

        real_cost = 0
        simu_cost = 0
        map_cost = self.hparams.compute_cost(predictions, clean_wavs, lens)

        # One is real, zero is fake
        if optim_name == "generator":
            simu_target = torch.ones(batch_size, 1)
            simu_cost = self.hparams.compute_cost(simu_result, simu_target)
            real_cost = 0.0
            self.metrics["G"].append(simu_cost.detach())
        elif optim_name == "discriminator":
            real_target = torch.ones(batch_size, 1)
            simu_target = torch.zeros(batch_size, 1)
            real_cost = self.hparams.compute_cost(real_result, real_target)
            simu_cost = self.hparams.compute_cost(simu_result, simu_target)
            self.metrics["D"].append((real_cost + simu_cost).detach())

        return real_cost + simu_cost + map_cost

    def fit_batch(self, batch):
        "Trains the GAN with a batch"
        self.g_optimizer.zero_grad()
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        g_loss = self.compute_objectives(
            predictions, batch, sb.Stage.TRAIN, "generator"
        )
        g_loss.backward()
        self.g_optimizer.step()

        self.d_optimizer.zero_grad()
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        d_loss = self.compute_objectives(
            predictions, batch, sb.Stage.TRAIN, "discriminator"
        )
        d_loss.backward()
        self.d_optimizer.step()

        return g_loss.detach() + d_loss.detach()

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage == sb.Stage.TRAIN:
            self.metrics = {"G": [], "D": []}

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            g_loss = torch.tensor(self.metrics["G"])
            d_loss = torch.tensor(self.metrics["D"])
            print("Avg G loss: %.2f" % torch.mean(g_loss))
            print("Avg D loss: %.2f" % torch.mean(d_loss))
            print("train loss: ", stage_loss)
        elif stage == sb.Stage.VALID:
            print("Completed epoch %d" % epoch)
            print("Valid loss: %.3f" % stage_loss)
        else:
            self.test_loss = stage_loss

    def init_optimizers(self):
        """Initializes the generator and discriminator optimizers"""
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )
        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )


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
    gan_brain = EnhanceGanBrain(modules=hparams["modules"], hparams=hparams)

    # Training/validation loop
    gan_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    gan_brain.evaluate(valid_data)

    # Check test loss (mse), train loss is GAN loss
    assert gan_brain.test_loss < 0.002


if __name__ == "__main__":
    main()


def test_loss():
    main()
