#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.data_io.data_io import read_wav, to_longTensor
from speechbrain.data_io.encoders import TextEncoder
from speechbrain.data_io.dataloader import SaveableDataLoader, collate_pad
from speechbrain.data_io.datasets import SegmentedDataset
from speechbrain.data_io.utils import replace_entries
import torch
from speechbrain.utils.data_utils import FuncPipeline


class seq2seqBrain(sb.Brain):
    def compute_forward(self, x, stage):
        wavs, wav_lens = x["waveforms"]
        # TEMPORARY PATCH:
        wav_lens = torch.Tensor([x[-1] for x in wav_lens]).to(wavs.device)
        phns, _ = x["bos_phns"]
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)
        x = self.modules.enc(feats)

        # Prepend bos token at the beginning
        y_in = sb.data_io.prepend_bos_token(phns, self.hparams.bos)
        e_in = self.modules.emb(y_in)
        h, w = self.modules.dec(e_in, x, wav_lens)
        logits = self.modules.lin(h)
        outputs = self.hparams.softmax(logits)

        if stage != sb.Stage.TRAIN:
            seq, _ = self.hparams.searcher(x, wav_lens)
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        if stage == sb.Stage.TRAIN:
            outputs = predictions
        else:
            outputs, seq = predictions
        ids = batch["id"]
        phns, phns_lens = batch["eos_phns"]
        # TEMPORARY PATCH:
        phns_lens = torch.Tensor([x[-1] for x in phns_lens]).to(phns.device)

        loss = self.hparams.compute_cost(outputs, phns, length=phns_lens)

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(ids, seq, phns, target_len=phns_lens)

        return loss

    def fit_batch(self, batch):
        preds = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, batch, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        out = self.compute_forward(batch, stage)
        loss = self.compute_objectives(out, batch, stage)
        return loss.detach()

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

    # this can be done in data prep, in experiments.py only the loading then it is ncessary
    # + adding bos and unknown
    label_encoder = TextEncoder()
    if not os.path.isfile("/tmp/label_enc.pkl"):
        label_encoder.fit([train_examples, dev_examples], "phones")
        label_encoder.save("/tmp/label_enc.pkl")
    else:
        label_encoder.load("/tmp/label_enc.pkl")
    label_encoder.add_bos_eos(
        bos_encoding=hparams["bos"], eos_encoding=hparams["eos"]
    )
    label_encoder.add_unkw(hparams["unknown_index"])
    print("Number of tokens in label encoder {}".format(len(label_encoder)))

    # here we define the pipelines for audio and the labels
    # for labels we have to distinct pipelines one adds eos another the bos
    wav_pip = FuncPipeline("waveforms", funcs=read_wav)
    bos_pip = FuncPipeline(
        "phones",
        "bos_phns",
        funcs=(
            label_encoder.prepend_bos,
            label_encoder.encode_int,
            to_longTensor,
        ),
    )
    eos_pip = FuncPipeline(
        "phones",
        "eos_phns",
        funcs=(
            label_encoder.append_eos,
            label_encoder.encode_int,
            to_longTensor,
        ),
    )

    train_data = SegmentedDataset(
        train_examples, data_transforms=[wav_pip, bos_pip, eos_pip]
    )
    valid_data = SegmentedDataset(
        dev_examples, data_transforms=[wav_pip, bos_pip, eos_pip]
    )
    test_data = SegmentedDataset(
        test_examples, data_transforms=[wav_pip, bos_pip, eos_pip]
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

    seq2seq_brain = seq2seqBrain(
        hparams["modules"], hparams["opt_class"], hparams
    )
    seq2seq_brain.fit(
        range(hparams["N_epochs"]), train_loader, valid_loader,
    )
    seq2seq_brain.evaluate(test_loader)

    # Check that model overfits for integration test
    assert seq2seq_brain.train_loss < 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
