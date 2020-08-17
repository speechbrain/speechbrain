#!/usr/bin/python
import speechbrain as sb


class AlignBrain(sb.core.Brain):
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
        sum_alpha_T = self.aligner(predictions, lens, phns, phn_lens, "forward")
        loss = -sum_alpha_T.sum()

        if stage != sb.core.Stage.TRAIN:
            viterbi_scores, alignments = self.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.core.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.core.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)

        if stage != sb.core.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
