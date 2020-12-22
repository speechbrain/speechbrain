#!/usr/bin/python
import os
import torch
import speechbrain as sb


class SpkIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        wavs, lens = batch.wav
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        x = self.modules.linear1(feats)
        x = self.hparams.activation(x)
        x = self.modules.linear2(x)
        x = torch.mean(x, dim=1, keepdim=True)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        predictions, lens = predictions
        loss = self.hparams.compute_cost(predictions, batch.spk_id_enc, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                batch.id, predictions, batch.spk_id_enc, lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(
                stage, "error: %.2f" % self.error_metrics.summarize("average")
            )


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    # Data loaders
    # Label encoder:
    encoder = hparams["label_encoder"]
    dsets = [hparams["train_data"], hparams["valid_data"], hparams["test_data"]]
    for dset in dsets:
        # Note: in the legacy format, strings always return a list
        encoder.update_from_didataset(dset, "spk_id", sequence_input=True)
    for dset in dsets:
        dset.add_dynamic_item(
            "spk_id_enc", encoder.encode_sequence_torch, "spk_id"
        )
        dset.set_output_keys(["id", "wav", "spk_id_enc"])

    spk_id_brain = SpkIdBrain(hparams["modules"], hparams["opt_class"], hparams)
    spk_id_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_data"],
        hparams["valid_data"],
        **hparams["loader_kwargs"],
    )
    spk_id_brain.evaluate(hparams["test_data"], **hparams["loader_kwargs"])

    # Check that model overfits for an integration test
    assert spk_id_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
