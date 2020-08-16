#!/usr/bin/python
import speechbrain as sb


class AlignBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)
        x = self.model(feats, init_params=init_params)
        x = self.lin(x, init_params)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        ids, phns, phn_lens = targets

        prev_alignments = self.aligner.get_prev_alignments(
            ids, predictions, lens, phns, phn_lens
        )
        loss = self.compute_cost(predictions, prev_alignments)

        if stage != "train":
            viterbi_scores, alignments = self.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )
            self.aligner.store_alignments(ids, alignments)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == "train":
            self.train_loss = stage_loss
        if stage == "valid":
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
            print("Valid loss: %.2f" % stage_loss)
            print("Recalculating and recording alignments...")
            self.evaluate(self.train_set)
