#!/usr/bin/python
import torch
import speechbrain as sb


class EnhanceGanBrain(sb.core.Brain):
    def compute_forward(self, x, init_params=False):
        id, wavs, lens = x

        noisy = self.add_noise(wavs, lens).unsqueeze(-1)
        enhanced = self.generator(noisy, init_params=init_params)

        if init_params:
            self.discriminator(enhanced, init_params=init_params)

        return enhanced

    def compute_objectives(
        self, predictions, targets, optimizer_modules=[], stage="train"
    ):
        id, clean_wavs, lens = targets
        batch_size = clean_wavs.size(0)

        # Average the predictions of each time step
        clean_wavs = clean_wavs.unsqueeze(-1)
        real_result = self.discriminator(clean_wavs).mean(dim=1)
        simu_result = self.discriminator(predictions).mean(dim=1)

        real_cost = 0
        simu_cost = 0
        map_cost = self.compute_cost(predictions, clean_wavs, lens)

        # One is real, zero is fake
        if "generator" in optimizer_modules:
            simu_target = torch.ones(batch_size, 1)
            simu_cost = self.compute_cost(simu_result, simu_target)
            real_cost = 0.0
            self.metrics["G"].append(simu_cost.detach())
        elif "discriminator" in optimizer_modules:
            real_target = torch.ones(batch_size, 1)
            simu_target = torch.zeros(batch_size, 1)
            real_cost = self.compute_cost(real_result, real_target)
            simu_cost = self.compute_cost(simu_result, simu_target)
            self.metrics["D"].append((real_cost + simu_cost).detach())

        return real_cost + simu_cost + map_cost

    def fit_batch(self, batch):
        inputs = batch[0]

        # Iterate optimizers and update
        for modules, optimizer in self.optimizers.items():
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, inputs, modules)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0]
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, inputs, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        if stage == "train":
            self.metrics = {"G": [], "D": []}

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            g_loss = torch.tensor(self.metrics["G"])
            d_loss = torch.tensor(self.metrics["D"])
            print("Avg G loss: %.2f" % torch.mean(g_loss))
            print("Avg D loss: %.2f" % torch.mean(d_loss))
            print("train loss: ", stage_loss)
        if stage == "valid":
            print("Completed epoch %d" % epoch)
            print("Valid loss: %.3f" % stage_loss)
