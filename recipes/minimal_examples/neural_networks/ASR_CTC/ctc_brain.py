#!/usr/bin/python
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode


class CTCBrain(sb.core.Brain):
    def compute_forward(self, x, stage=sb.core.Stage.TRAIN, init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)
        x = self.model(feats, init_params=init_params)
        x = self.lin(x, init_params)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(
        self, predictions, targets, stage=sb.core.Stage.TRAIN
    ):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        loss = self.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.core.Stage.TRAIN:
            seq = ctc_greedy_decode(predictions, lens, blank_id=-1)
            self.per_metrics.append(ids, seq, phns, phn_lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.core.Stage.TRAIN:
            self.per_metrics = self.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.core.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage == sb.core.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)

        if stage != sb.core.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))
