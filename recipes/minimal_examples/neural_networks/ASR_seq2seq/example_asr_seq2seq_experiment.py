#!/usr/bin/python
import os
import speechbrain as sb


class seq2seqBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
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
        preds = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
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

    # Update label encoder:
    label_encoder = hparams["label_encoder"]
    label_encoder.update_from_didataset(
        hparams["train_data"], output_key="phn_list", sequence_input=True
    )
    label_encoder.update_from_didataset(
        hparams["valid_data"], output_key="phn_list", sequence_input=True
    )
    label_encoder.insert_bos_eos(bos_index=hparams["eos_bos_index"])
    label_encoder.insert_blank(index=hparams["blank_index"])

    seq2seq_brain = seq2seqBrain(
        hparams["modules"], hparams["opt_class"], hparams
    )
    seq2seq_brain.fit(
        range(hparams["N_epochs"]),
        hparams["train_data"],
        hparams["valid_data"],
    )
    seq2seq_brain.evaluate(hparams["valid_data"])

    # Check that model overfits for integration test
    assert seq2seq_brain.train_loss < 1.0


if __name__ == "__main__":
    main()


def test_error():
    main()
