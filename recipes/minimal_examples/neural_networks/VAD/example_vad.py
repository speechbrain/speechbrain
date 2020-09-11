#!/usr/bin/python
import os
import torch
import numpy as np
import speechbrain as sb


class VADBrain(sb.Brain):
    def compute_forward(self, x, stage):
        id, wavs, lens = x
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, lens)
        x = self.rnn(feats)
        outputs = self.lin(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage=True):
        predictions, lens = predictions

        ids, targets, lens = targets
        targets = targets.to(predictions.device)
        predictions = predictions[:, : targets.shape[-1], 0]
        loss = self.compute_BCE_cost(
            torch.nn.BCEWithLogitsLoss(reduction="none"),
            predictions,
            targets,
            lens,
        )

        # compute metrics
        self.binary_metrics.append(ids, torch.sigmoid(predictions), targets)

        return loss

    def on_stage_start(self, stage, epoch=None):
        self.binary_metrics = sb.BinaryMetricStats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            train_summary = self.binary_metrics.summarize(threshold=0.5)

            print("Epoch %d completed" % epoch)
            print("Train loss: %.4f" % stage_loss)
            print("Train Precision: %.2f" % train_summary["precision"])
            print("Train Recall: %.2f" % train_summary["recall"])


def parsing_func(params, string):
    boundaries = string.split(" ")
    # we group by two
    # 0.01 is 10 ms hop size ...
    # IS THERE AN EASY WAY to pass to mfcc the step size ?
    boundaries = [int(float(x) / 0.01) for x in boundaries]
    boundaries = list(zip(boundaries[::2], boundaries[1::2]))

    gt = torch.zeros(int(np.ceil(params.example_length * (1 / 0.01))))

    for indxs in boundaries:
        start, stop = indxs
        gt[start:stop] = 1

    return gt


def main():
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(experiment_dir, "hyperparams.yaml")
    data_folder = "../../../../../samples/vad"
    data_folder = os.path.abspath(experiment_dir + data_folder)
    with open(params_file) as fin:
        params = sb.load_extended_yaml(fin, {"data_folder": data_folder})

    train_set = sb.DataLoaderFactory(
        params.csv_train,
        params.N_batch,
        ["wav", "speech"],
        replacements={"$data_folder": params.data_folder},
        label_parsing_func=lambda x: parsing_func(params, x),
    )()

    vad_brain = VADBrain(
        modules=params.modules, optimizers=["optimizer"], device="cpu",
    )
    vad_brain.fit(range(params.N_epochs), train_set)

    assert vad_brain.train_loss < 0.01


if __name__ == "__main__":
    main()


def test_error():
    main()
