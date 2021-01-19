#!/usr/bin/env/python3
"""This minimal example trains a seq2seq attention-based model for speech
recognition on a tiny dataset.  The encoder is based on a combination of
convolutional, recurrent, and feed-forward networks (CRDNN). The decoder is
based on a GRU. A greedy search  is used on top of the output probabilities.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).
"""
import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class seq2seqBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)
        x = self.modules.enc(feats)

        # Prepend bos token at the beginning
        e_in = self.modules.emb(phns_bos)
        h, w = self.modules.dec(e_in, x, wav_lens)
        logits = self.modules.lin(h)
        outputs = self.hparams.softmax(logits)

        if stage != sb.Stage.TRAIN:
            seq, _ = self.hparams.searcher(x, wav_lens)
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if stage == sb.Stage.TRAIN:
            outputs = predictions
        else:
            outputs, seq = predictions

        ids = batch.id
        phns, phn_lens = batch.phn_encoded_eos

        loss = self.hparams.compute_cost(outputs, phns, length=phn_lens)

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(ids, seq, phns, target_len=phn_lens)

        return loss

    def fit_batch(self, batch):
        """Fits train batches"""
        preds = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        """Evaluates test batches"""
        out = self.compute_forward(batch, stage)
        loss = self.compute_objectives(out, batch, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        "Gets called when a stage (either training, validation, test) ends."
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "train.json",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "dev.json",
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list", "phn_encoded_bos", "phn_encoded_eos"
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        phn_encoded_bos = label_encoder.prepend_bos_index(phn_encoded).long()
        yield phn_encoded_bos
        phn_encoded_eos = label_encoder.append_eos_index(phn_encoded).long()
        yield phn_encoded_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.insert_bos_eos(bos_index=hparams["bos_index"])
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    label_encoder.update_from_didataset(valid_data, output_key="phn_list")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "phn_encoded_eos", "phn_encoded_bos"]
    )
    return train_data, valid_data


def main():
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = "../../../../samples/audio_samples/nn_training_samples"
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    seq2seq_brain = seq2seqBrain(
        hparams["modules"], hparams["opt_class"], hparams
    )

    # Training/validation loop
    seq2seq_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    seq2seq_brain.evaluate(valid_data)

    # Check that model overfits for integration test
    assert seq2seq_brain.train_loss < 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
