#!/usr/bin/python
import os
import speechbrain as sb


class ASR_Brain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        x = self.modules.linear1(feats)
        x = self.hparams.activation(x)
        x = self.modules.linear2(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage):
        outputs, lens = predictions
        ids, ali, ali_lens = targets
        loss = self.hparams.compute_cost(outputs, ali, lens)

        if stage != sb.Stage.TRAIN:
            self.err_metrics.append(ids, outputs, ali, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.err_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)

        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "error: %.2f" % self.err_metrics.summarize("average"))


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    asr_brain = ASR_Brain(hparams["modules"], hparams["opt_class"], hparams)
    asr_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_loader"](),
        hparams["valid_loader"](),
    )
    asr_brain.evaluate(hparams["test_loader"]())

    # Check that model overfits for integration test
    assert asr_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
