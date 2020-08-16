#!/usr/bin/python
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode


class CTCBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)

        outputs = self.model(feats, init_params)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        loss = self.compute_cost(predictions, phns, lens, phn_lens)

        if stage != "train":
            seq = ctc_greedy_decode(predictions, lens, blank_id=-1)
            self.per_metrics.append(ids, seq, phns, phn_lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != "train":
            self.per_metrics = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            self.train_loss = stage_loss
        if stage == "valid":
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != "train":
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize())
