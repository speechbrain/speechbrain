#!/usr/bin/python
import torch
import speechbrain as sb


class SpkIdBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)

        x = self.linear1(feats, init_params)
        x = self.activation(x)
        x = self.linear2(x, init_params)
        x = torch.mean(x, dim=1, keepdim=True)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets
        loss = self.compute_cost(predictions, spkid, lens)

        if stage != "train":
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != "train":
            self.error_metrics = self.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            self.train_loss = stage_loss
        if stage == "valid":
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != "train":
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "error: %.2f" % self.error_metrics.summarize())
