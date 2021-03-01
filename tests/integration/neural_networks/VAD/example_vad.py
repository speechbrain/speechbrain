"""This minimal example trains a Voice Activity Detector (VAD) on a tiny dataset.
The network is based on a LSTM with a linear transformation on the top of that.
The system is trained with the binary cross-entropy metric.
"""

import os
import torch
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class VADBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the binary probability."
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x, _ = self.modules.rnn(feats)
        outputs = self.modules.lin(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage=True):
        "Given the network predictions and targets computed the binary CE"
        predictions, lens = predictions

        targets, lens = batch.target
        targets = targets.to(predictions.device)
        predictions = predictions[:, : targets.shape[-1], 0]
        loss = self.hparams.compute_BCE_cost(
            torch.nn.BCEWithLogitsLoss(reduction="none"),
            predictions,
            targets,
            lens,
        )

        # compute metrics
        self.binary_metrics.append(
            batch.id, torch.sigmoid(predictions), targets
        )

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        self.binary_metrics = sb.utils.metric_stats.BinaryMetricStats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            train_summary = self.binary_metrics.summarize(threshold=0.5)

            print("Epoch %d completed" % epoch)
            print("Train loss: %.4f" % stage_loss)
            print("Train Precision: %.2f" % train_summary["precision"])
            print("Train Recall: %.2f" % train_summary["recall"])


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=os.path.join(data_folder, "train.json"),
        replacements={"data_folder": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=os.path.join(data_folder, "valid.json"),
        replacements={"data_folder": data_folder},
    )

    datasets = [train_data, valid_data]

    # 1. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 2. vad targets creation from annotated speech boundaries
    @sb.utils.data_pipeline.takes("speech")
    @sb.utils.data_pipeline.provides("target")
    def vad_targets(string, hparams=hparams):
        if len(string) > 0:
            boundaries = string.split(" ")
            # we group by two
            # 0.01 is 10 ms hop size ...
            boundaries = [int(float(x) / 0.01) for x in boundaries]
            boundaries = list(zip(boundaries[::2], boundaries[1::2]))
        else:
            boundaries = []

        gt = torch.zeros(int(np.ceil(hparams["example_length"] * (1 / 0.01))))

        for indxs in boundaries:
            start, stop = indxs
            gt[start:stop] = 1

        return gt

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, vad_targets)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "target"])

    return train_data, valid_data


def main():

    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    hparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../../samples/audio_samples/vad"
    data_folder = os.path.abspath(experiment_dir + data_folder)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Data IO creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    ctc_brain = VADBrain(hparams["modules"], hparams["opt_class"], hparams)

    # Training/validation loop
    ctc_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    ctc_brain.evaluate(valid_data)

    # Check if model overfits for integration test
    assert ctc_brain.train_loss < 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
