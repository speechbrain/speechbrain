#!/usr/bin/python
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average


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

        stats = {}

        if stage != "train":
            viterbi_scores, alignments = self.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )

            self.aligner.store_alignments(ids, alignments)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))

        print("Recalculating and recording alignments...")
        self.evaluate(self.train_set)
