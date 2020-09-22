#!/usr/bin/python
import os
import torch
import speechbrain as sb


class EnhanceGanBrain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x

        noisy = self.hparams.add_noise(wavs, lens).unsqueeze(-1)
        enhanced = self.hparams.generator(noisy)

        return enhanced

    def compute_objectives(self, predictions, targets, stage, optim_name=""):
        id, clean_wavs, lens = targets
        batch_size = clean_wavs.size(0)

        # Average the predictions of each time step
        clean_wavs = clean_wavs.unsqueeze(-1)
        real_result = self.hparams.discriminator(clean_wavs).mean(dim=1)
        simu_result = self.hparams.discriminator(predictions).mean(dim=1)

        real_cost = 0
        simu_cost = 0
        map_cost = self.hparams.compute_cost(predictions, clean_wavs, lens)

        # One is real, zero is fake
        if optim_name == "g_optimizer":
            simu_target = torch.ones(batch_size, 1)
            simu_cost = self.hparams.compute_cost(simu_result, simu_target)
            real_cost = 0.0
            self.metrics["G"].append(simu_cost.detach())
        elif optim_name == "d_optimizer":
            real_target = torch.ones(batch_size, 1)
            simu_target = torch.zeros(batch_size, 1)
            real_cost = self.hparams.compute_cost(real_result, real_target)
            simu_cost = self.hparams.compute_cost(simu_result, simu_target)
            self.metrics["D"].append((real_cost + simu_cost).detach())

        return real_cost + simu_cost + map_cost

    def fit_batch(self, batch):
        inputs = batch[0]

        # Iterate optimizers and update
        for optim_name, optimizer in self.optim.__dict__.items():
            predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
            loss = self.compute_objectives(
                predictions, inputs, sb.Stage.TRAIN, optim_name
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        inputs = batch[0]
        predictions = self.compute_forward(inputs, stage)
        loss = self.compute_objectives(predictions, inputs, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.metrics = {"G": [], "D": []}

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            g_loss = torch.tensor(self.metrics["G"])
            d_loss = torch.tensor(self.metrics["D"])
            print("Avg G loss: %.2f" % torch.mean(g_loss))
            print("Avg D loss: %.2f" % torch.mean(d_loss))
            print("train loss: ", stage_loss)
        if stage == sb.Stage.VALID:
            print("Completed epoch %d" % epoch)
            print("Valid loss: %.3f" % stage_loss)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    auto_brain = EnhanceGanBrain(hparams["hparams"], hparams["optim"])
    auto_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_loader"](),
        hparams["valid_loader"](),
    )
    test_loss = auto_brain.evaluate(hparams["test_loader"]())

    # Check test loss (mse), train loss is GAN loss
    assert test_loss < 0.002


if __name__ == "__main__":
    main()


def test_loss():
    main()
