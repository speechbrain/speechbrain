#!/usr/bin/env/python3
"""This minimal example trains a RNNT-based speech recognizer on a tiny dataset.
The encoder is based on a Conformer model with the use of Dynamic Chunk Training
 (with a Dynamic Chunk Convolution within the convolution modules) that predict
phonemes. A greedy search is used on top of the output probabilities.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).
"""
import pathlib

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb


class ConformerTransducerBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phn_with_bos, phn_with_bos_lens = batch.phn_encoded_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
                phn_with_bos = self.hparams.wav_augment.replicate_labels(
                    phn_with_bos
                )

        feats = self.hparams.compute_features(wavs)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            phn_with_bos = self.hparams.fea_augment.replicate_labels(
                phn_with_bos
            )

        current_epoch = self.hparams.epoch_counter.current

        # Old models may not have the streaming hparam, we don't break them in
        # any other way so just check for its presence
        if hasattr(self.hparams, "streaming") and self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
        else:
            dynchunktrain_config = None

        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        src = self.modules.CNN(feats)
        x = self.modules.enc(
            src,
            wav_lens,
            pad_idx=self.hparams.pad_index,
            dynchunktrain_config=dynchunktrain_config,
        )
        x = self.modules.proj_enc(x)

        e_in = self.modules.emb(phn_with_bos)
        e_in = torch.nn.functional.dropout(
            e_in,
            self.hparams.dec_emb_dropout,
            training=(stage == sb.Stage.TRAIN),
        )
        h, _ = self.modules.dec(e_in)
        h = torch.nn.functional.dropout(
            h, self.hparams.dec_dropout, training=(stage == sb.Stage.TRAIN)
        )
        h = self.modules.proj_dec(h)

        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits_transducer = self.modules.transducer_lin(joint)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            p_ctc = None
            p_ce = None

            if self.hparams.ctc_weight > 0.0:
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.proj_ctc(x)
                p_ctc = self.hparams.log_softmax(out_ctc)

            if self.hparams.ce_weight > 0.0:
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)

            return p_ctc, p_ce, logits_transducer, wav_lens

        best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
        return logits_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""
        ids = batch.id
        phn, phn_lens = batch.phn_encoded
        phn_with_eos, phn_with_eos_lens = batch.phn_encoded_eos

        # Train returns 4 elements vs 3 for val and test
        if len(predictions) == 4:
            p_ctc, p_ce, logits_transducer, wav_lens = predictions
        else:
            logits_transducer, wav_lens, predicted_phn = predictions

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                phn = self.hparams.wav_augment.replicate_labels(phn)
                phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)
                phn_with_eos = self.hparams.wav_augment.replicate_labels(
                    phn_with_eos
                )
                phn_with_eos_lens = self.hparams.wav_augment.replicate_labels(
                    phn_with_eos_lens
                )
            if hasattr(self.hparams, "fea_augment"):
                phn = self.hparams.fea_augment.replicate_labels(phn)
                phn_lens = self.hparams.fea_augment.replicate_labels(phn_lens)
                phn_with_eos = self.hparams.fea_augment.replicate_labels(
                    phn_with_eos
                )
                phn_with_eos_lens = self.hparams.fea_augment.replicate_labels(
                    phn_with_eos_lens
                )

        if stage == sb.Stage.TRAIN:
            CTC_loss = 0.0
            CE_loss = 0.0
            if p_ctc is not None:
                CTC_loss = self.hparams.ctc_cost(p_ctc, phn, wav_lens, phn_lens)
            if p_ce is not None:
                CE_loss = self.hparams.ce_cost(
                    p_ce, phn_with_eos, length=phn_with_eos_lens
                )
            loss_transducer = self.hparams.transducer_cost(
                logits_transducer, phn, wav_lens, phn_lens
            )
            loss = (
                self.hparams.ctc_weight * CTC_loss
                + self.hparams.ce_weight * CE_loss
                + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                * loss_transducer
            )
        else:
            loss = self.hparams.transducer_cost(
                logits_transducer, phn, wav_lens, phn_lens
            )

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(
                ids, predicted_phn, phn, target_len=phn_lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called when a stage (either training, validation, test) starts."""
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))


def data_prep(data_folder, hparams):
    """Creates the datasets and their data processing pipelines."""
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "../annotation/ASR_train.json",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "../annotation/ASR_dev.json",
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    label_encoder.expect_len(hparams["num_labels"])

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
        "phn_list", "phn_encoded", "phn_encoded_bos", "phn_encoded_eos"
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded
        phn_encoded_bos = label_encoder.prepend_bos_index(phn_encoded).long()
        yield phn_encoded_bos
        phn_encoded_eos = label_encoder.append_eos_index(phn_encoded).long()
        yield phn_encoded_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.insert_blank(index=hparams["blank_index"])
    label_encoder.insert_bos_eos(
        bos_index=hparams["bos_index"], eos_label="<bos>"
    )
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    label_encoder.update_from_didataset(valid_data, output_key="phn_list")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_encoded_bos", "phn_encoded_eos"],
    )
    return train_data, valid_data, label_encoder


def main(device="cpu"):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"
    data_folder = "../../samples/ASR"
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data, label_encoder = data_prep(data_folder, hparams)

    # Trainer initialization
    transducer_brain = ConformerTransducerBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
    )

    # Training/validation loop
    transducer_brain.fit(
        range(hparams["number_of_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    transducer_brain.evaluate(valid_data)

    # Check that model overfits for integration test
    assert transducer_brain.train_loss < 90.0


if __name__ == "__main__":
    main()


def test_error(device):
    main(device)
