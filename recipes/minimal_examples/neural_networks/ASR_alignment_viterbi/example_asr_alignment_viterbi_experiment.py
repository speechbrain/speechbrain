#!/usr/bin/python
import os
import speechbrain as sb


class AlignBrain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, lens)
        x = self.model(feats)
        x = self.lin(x)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage):
        predictions, lens = predictions
        ids, phns, phn_lens = targets

        prev_alignments = self.aligner.get_prev_alignments(
            ids, predictions, lens, phns, phn_lens
        )
        loss = self.compute_cost(predictions, prev_alignments)

        if stage != sb.Stage.TRAIN:
            viterbi_scores, alignments = self.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )
            self.aligner.store_alignments(ids, alignments)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
            print("Valid loss: %.2f" % stage_loss)
            print("Recalculating and recording alignments...")
            self.evaluate(self.train_set)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    train_set = hyperparams.train_loader()
    hyperparams.modules["train_set"] = train_set
    align_brain = AlignBrain(
        modules=hyperparams.modules,
        optimizers={("model", "lin"): hyperparams.optimizer},
        device="cpu",
    )
    align_brain.fit(
        range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
    )
    align_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for integration test
    assert align_brain.train_loss < 2.0


if __name__ == "__main__":
    main()


def test_error():
    main()
