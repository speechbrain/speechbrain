#!/usr/bin/python
import os
import speechbrain as sb


class ASR_Brain(sb.Brain):
    def compute_forward(self, x, stage=sb.Stage.TRAIN, init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)

        x = self.linear1(feats, init_params)
        x = self.activation(x)
        x = self.linear2(x, init_params)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage=sb.Stage.TRAIN):
        outputs, lens = predictions
        ids, ali, ali_lens = targets
        loss = self.compute_cost(outputs, ali, lens)

        if stage != sb.Stage.TRAIN:
            self.err_metrics.append(ids, outputs, ali, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.err_metrics = self.error_stats()

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
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    train_set = hyperparams.train_loader()
    first_x, first_y = next(iter(train_set))
    asr_brain = ASR_Brain(
        modules=hyperparams.modules,
        optimizers={("linear1", "linear2"): hyperparams.optimizer},
        device="cpu",
        first_inputs=[first_x],
    )
    asr_brain.fit(
        range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
    )
    asr_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for integration test
    assert asr_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
