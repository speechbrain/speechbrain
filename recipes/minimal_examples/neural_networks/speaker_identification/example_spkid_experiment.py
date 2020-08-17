#!/usr/bin/python
import os
import torch
import speechbrain as sb


class SpkIdBrain(sb.core.Brain):
    def compute_forward(self, x, stage=sb.core.Stage.TRAIN, init_params=False):
        id, wavs, lens = x
        feats = self.compute_features(wavs, init_params)
        feats = self.mean_var_norm(feats, lens)

        x = self.linear1(feats, init_params)
        x = self.activation(x)
        x = self.linear2(x, init_params)
        x = torch.mean(x, dim=1, keepdim=True)
        outputs = self.softmax(x)

        return outputs, lens

    def compute_objectives(
        self, predictions, targets, stage=sb.core.Stage.TRAIN
    ):
        predictions, lens = predictions
        uttid, spkid, _ = targets
        loss = self.compute_cost(predictions, spkid, lens)

        if stage != sb.core.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.core.Stage.TRAIN:
            self.error_metrics = self.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.core.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.core.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.core.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(
                stage, "error: %.2f" % self.error_metrics.summarize("average")
            )


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hyperparams_file) as fin:
        hyperparams = sb.yaml.load_extended_yaml(
            fin, {"data_folder": data_folder}
        )

    train_set = hyperparams.train_loader()
    first_x, first_y = next(iter(train_set))
    spk_id_brain = SpkIdBrain(
        modules=hyperparams.modules,
        optimizers={("linear1", "linear2"): hyperparams.optimizer},
        device="cpu",
        first_inputs=[first_x],
    )
    spk_id_brain.fit(
        range(hyperparams.N_epochs), train_set, hyperparams.valid_loader()
    )
    spk_id_brain.evaluate(hyperparams.test_loader())

    # Check that model overfits for an integration test
    assert spk_id_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
