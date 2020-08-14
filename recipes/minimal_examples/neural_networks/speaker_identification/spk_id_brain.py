#!/usr/bin/python
import torch
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average


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

        stats = {}

        if stage != "train":
            stats["error"] = self.compute_error(predictions, spkid, lens)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid error: %.2f" % summarize_average(valid_stats["error"]))
