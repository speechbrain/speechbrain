#!/usr/bin/python
import os
import speechbrain as sb


class AlignBrain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        sum_alpha_T = self.hparams.aligner(
            predictions, lens, phns, phn_lens, "forward"
        )
        loss = -sum_alpha_T.sum()

        if stage != sb.Stage.TRAIN:
            viterbi_scores, alignments = self.hparams.aligner(
                predictions, lens, phns, phn_lens, "viterbi"
            )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)

        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    align_brain = AlignBrain(hparams["modules"], hparams["opt_class"], hparams)
    align_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_loader"](),
        hparams["valid_loader"](),
    )
    align_brain.evaluate(hparams["test_loader"]())

    # Check that model overfits for integration test
    assert align_brain.train_loss < 300.0


if __name__ == "__main__":
    main()


def test_error():
    main()
