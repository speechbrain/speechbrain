#!/usr/bin/python
import os
import speechbrain as sb
import torch

from speechbrain.data_io.utils import replace_entries

# TODO: Replace local placeholder Dataset class
from speechbrain.data_io.datasets import SegmentedDataset

# TODO: Replace ASR Dataset transforms for now no FuncPipeline:
# from placeholders import FuncPipeline
from speechbrain.data_io.data_io import read_wav
from speechbrain.data_io.encoders import TextEncoder

# from placeholders import split_by_whitespace
# from placeholders import ExampleCategoricalEncoder
# from placeholders import to_int_tensor

# TODO: Replace label dict creation
# from placeholders import ASR_example_label2ind, ASR_example_ind2label

# TODO: Replace collate fn:
from speechbrain.data_io.dataloader import collate_pad
from speechbrain.data_io.dataloader import SaveableDataLoader


class CTCBrain(sb.Brain):
    def compute_forward(self, x, stage):

        wavs, lens = x["waveforms"]
        # TEMPORARY PATCH:
        lens = torch.Tensor([x[-1] for x in lens]).to(wavs.device)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):

        predictions, lens = predictions
        phns, phn_lens = batch["phones"]
        ids = batch["id"]
        # SAME UGLY PATCHES AS BEFORE:
        phn_lens = torch.Tensor([x[-1] for x in phn_lens]).to(
            predictions.device
        )
        loss = self.hparams.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            seq = sb.decoders.ctc_greedy_decode(
                predictions, lens, blank_id=self.hparams.blank_index
            )
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
        hparams = sb.yaml.load_extended_yaml(fin)

    with open(os.path.join(data_folder, "train.yaml"), "r",) as f:
        train_examples = sb.yaml.load_extended_yaml(f)
    with open(os.path.join(data_folder, "dev.yaml"), "r",) as f:
        dev_examples = sb.yaml.load_extended_yaml(f)
    with open(os.path.join(data_folder, "test.yaml"), "r",) as f:
        test_examples = sb.yaml.load_extended_yaml(f)

    # TODO this can be handled into load_extended yaml in future but requires modifications
    # to core extended yaml overriding functionality.
    replacements_dict = {
        "files": {"DATASET_ROOT": data_folder},
        "alignment_file": {"ALIGNMENTS_ROOT": data_folder},
    }

    replace_entries(train_examples, replacements_dict)  # in place substitution
    replace_entries(dev_examples, replacements_dict)
    replace_entries(test_examples, replacements_dict)

    label_encoder = TextEncoder()
    if not os.path.isfile("/tmp/label_enc.pkl"):
        label_encoder.fit([train_examples, dev_examples], "phones")
        label_encoder.save("/tmp/label_enc.pkl")
    else:
        label_encoder.load("/tmp/label_enc.pkl")
    label_encoder.add_blank(hparams["blank_index"], "<blank>")
    label_encoder.add_unkw(hparams["unknown_index"], "<unknown>")

    # TODO: Convert minimal example CSV to new YAML format
    train_data = SegmentedDataset(
        train_examples,
        ["waveforms", "phones"],
        {"waveforms": read_wav, "phones": label_encoder.encode_int},
    )
    valid_data = SegmentedDataset(
        dev_examples,
        ["waveforms", "phones"],
        {"waveforms": read_wav, "phones": label_encoder.encode_int},
    )
    test_data = SegmentedDataset(
        test_examples,
        ["waveforms", "phones"],
        {"waveforms": read_wav, "phones": label_encoder.encode_int},
    )

    # Placeholders:
    train_loader = SaveableDataLoader(
        train_data, batch_size=hparams["N_batch"], collate_fn=collate_pad,
    )
    valid_loader = SaveableDataLoader(
        valid_data, batch_size=1, collate_fn=collate_pad
    )
    test_loader = SaveableDataLoader(
        test_data, batch_size=1, collate_fn=collate_pad
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
