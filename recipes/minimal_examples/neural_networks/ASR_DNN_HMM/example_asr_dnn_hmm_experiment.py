#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.data_io.data_io import read_wav, read_pkl
from speechbrain.data_io.encoders import TextEncoder
from speechbrain.data_io.dataloader import SaveableDataLoader, collate_pad
from speechbrain.data_io.datasets import SegmentedDataset
from speechbrain.data_io.utils import replace_entries
import torch
from speechbrain.utils.data_utils import FuncPipeline


class ASR_Brain(sb.Brain):
    def compute_forward(self, x, stage):
        wavs, lens = x["waveforms"]
        # TEMPORARY PATCH:
        lens = torch.Tensor([x[-1] for x in lens]).to(wavs.device)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        x = self.modules.linear1(feats)
        x = self.hparams.activation(x)
        x = self.modules.linear2(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        outputs, lens = predictions
        ali, ali_lens = batch["alignment_file"]
        ids = batch["id"]
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

    wav_pip = FuncPipeline("waveforms", funcs=read_wav)
    phn_pip = FuncPipeline("alignment_file", funcs=read_pkl)
    # TODO: Convert minimal example CSV to new YAML format
    train_data = SegmentedDataset(
        train_examples, data_transforms=[wav_pip, phn_pip]
    )
    valid_data = SegmentedDataset(
        dev_examples, data_transforms=[wav_pip, phn_pip]
    )
    test_data = SegmentedDataset(
        test_examples, data_transforms=[wav_pip, phn_pip]
    )

    train_loader = SaveableDataLoader(
        train_data, batch_size=hparams["N_batch"], collate_fn=collate_pad,
    )
    valid_loader = SaveableDataLoader(
        valid_data, batch_size=1, collate_fn=collate_pad
    )
    test_loader = SaveableDataLoader(
        test_data, batch_size=1, collate_fn=collate_pad
    )

    asr_brain = ASR_Brain(hparams["modules"], hparams["opt_class"], hparams)
    asr_brain.fit(
        range(hparams["N_epochs"]), train_loader, valid_loader,
    )
    asr_brain.evaluate(test_loader)

    # Check that model overfits for integration test
    assert asr_brain.train_loss < 0.2


if __name__ == "__main__":
    main()


def test_error():
    main()
