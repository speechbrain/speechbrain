#!/usr/bin/python
import os
import speechbrain as sb


class AlignBrain(sb.Brain):
    def compute_forward(self, x, stage=sb.Stage.TRAIN, init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)
        x = self.model(feats, init_params=init_params)
        x = self.lin(x, init_params)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage=sb.Stage.TRAIN):
        predictions, lens = predictions
        ids, phns, phn_lens = targets
        sum_alpha_T = self.aligner(predictions, lens, phns, phn_lens, "forward")
        loss = -sum_alpha_T.sum()

        if stage != sb.Stage.TRAIN:
            viterbi_scores, alignments = self.aligner(
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
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    train_set = hyperparams.train_loader()
    first_x, first_y = next(iter(train_set))
    align_brain = AlignBrain(
        modules=hyperparams.modules,
        optimizers={("model", "lin"): hyperparams.optimizer},
        device="cpu",
        first_inputs=[first_x],
    )
    align_brain.fit(
        range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
    )
    align_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for integration test
    assert align_brain.train_loss < 300.0


if __name__ == "__main__":
    main()


def test_error():
    main()
