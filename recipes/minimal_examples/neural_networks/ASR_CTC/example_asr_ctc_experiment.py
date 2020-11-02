#!/usr/bin/python
import os
import speechbrain as sb

# TODO: Replace local placeholder Dataset class
from placeholders import ASRMinimalExampleDataset

# TODO: Replace ASR Dataset transforms:
from placeholders import torchaudio_load

# TODO: Replace collate fn:
from placeholders import ASR_example_collation
from speechbrain.data_io.dataloader import SaveableDataLoader


class CTCBrain(sb.Brain):
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
        loss = self.hparams.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            seq = sb.decoders.ctc_greedy_decode(predictions, lens, blank_id=-1)
            self.per_metrics.append(ids, seq, phns, target_len=phn_lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)

        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))


def main():
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
    with open(hparams_file) as fin:
        # TODO: Data loading back into YAML:
        hparams = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

        valid_data = ASRMinimalExampleDataset(
            os.path.join(data_folder, "dev.csv"),
            audio_transform=torchaudio_load,
            text_transform=hparams["text_transform"],
        )
        test_data = ASRMinimalExampleDataset(
            os.path.join(data_folder, "dev.csv"),
            audio_transform=torchaudio_load,
            text_transform=hparams["text_transform"],
        )

    # Placeholders:
    train_loader = SaveableDataLoader(
        hparams["train_data"],
        batch_size=hparams["N_batch"],
        collate_fn=ASR_example_collation,
    )
    valid_loader = SaveableDataLoader(
        valid_data, batch_size=1, collate_fn=ASR_example_collation
    )
    test_loader = SaveableDataLoader(
        test_data, batch_size=1, collate_fn=ASR_example_collation
    )

    ctc_brain = CTCBrain(hparams["modules"], hparams["opt_class"], hparams)
    ctc_brain.fit(
        range(hparams["N_epochs"]), train_loader, valid_loader,
    )
    ctc_brain.evaluate(test_loader)

    # Check if model overfits for integration test
    assert ctc_brain.train_loss < 3.0


if __name__ == "__main__":
    main()


def test_error():
    main()
