#!/usr/bin/python
import os
import torch
import speechbrain as sb


class EnhanceGanBrain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x

        noisy = self.hparams.add_noise(wavs, lens).unsqueeze(-1)
        enhanced = self.modules.generator(noisy)

        return enhanced

    def compute_objectives(self, predictions, targets, stage, optim_name=""):
        id, clean_wavs, lens = targets
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
        inputs = batch[0]

        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        g_loss = self.compute_objectives(
            predictions, inputs, sb.Stage.TRAIN, "generator"
        )
        g_loss.backward()
        self.g_optimizer.step()
        self.g_optimizer.zero_grad()

        predictions = self.compute_forward(inputs, sb.Stage.TRAIN)
        d_loss = self.compute_objectives(
            predictions, inputs, sb.Stage.TRAIN, "discriminator"
        )
        d_loss.backward()
        self.d_optimizer.step()
        self.d_optimizer.zero_grad()

        return g_loss.detach() + d_loss.detach()

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
        elif stage == sb.Stage.VALID:
            print("Completed epoch %d" % epoch)
            print("Valid loss: %.3f" % stage_loss)
        else:
            self.test_loss = stage_loss

    def init_optimizers(self):
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )
        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    gan_brain = EnhanceGanBrain(hparams["modules"], hparams=hparams)
    gan_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_loader"](),
        hparams["valid_loader"](),
    )
    gan_brain.evaluate(hparams["test_loader"]())

    # Check test loss (mse), train loss is GAN loss
    assert gan_brain.test_loss < 0.002


if __name__ == "__main__":
    main()


def test_loss():
    main()
