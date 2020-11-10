#!/usr/bin/env/python3
"""Recipe for doing ASR with phoneme targets with
Transducer loss on the TIMIT dataset.

To run this recipe, do the following:
> python experiment.py {hyperparameter file} --data_folder /path/to/TIMIT

Using your own hyperparameter file or one of the following:
 * hyperparams/augment_CRDNN.yaml
 * hyperparams/augment_noise_CRDNN.yaml

Authors
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
"""
import os
import sys
import torch
import speechbrain as sb


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, y, stage):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])
            phns = torch.cat([phns, phns])

        if hasattr(self.modules, "augmentation"):
            wavs = self.modules.augmentation(wavs, wav_lens)
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)
        x = self.modules.enc_lin(x)

        # Prepend bos token at the beginning
        y_in = sb.data_io.data_io.prepend_bos_token(
            phns, self.hparams.blank_index
        )
        e_in = self.modules.emb(y_in)
        h, _ = self.modules.dec(e_in)
        h = self.modules.dec_lin(h)

        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # output layer for seq2seq log-probabilities
        logits = self.modules.output(joint)
        p_transducer = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            hyps, scores, _, _ = self.Greedysearcher(x)
            return p_transducer, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.Beamsearcher(x)
            return p_transducer, wav_lens, best_hyps
        return p_transducer, wav_lens

    def compute_objectives(self, predictions, targets, stage):
        if stage == sb.Stage.TRAIN:
            p_transducer, wav_lens = predictions
        else:
            p_transducer, wav_lens, hyps = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if hasattr(self.hparams, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
        phns = phns.long()
        loss = self.hparams.compute_cost(p_transducer, phns, wav_lens, phn_lens)

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            self.transducer_metrics.append(
                ids, p_transducer, phns, wav_lens, phn_lens
            )
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.hparams.ind2lab
            )

        return loss

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss = self.compute_objectives(predictions, targets, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        self.transducer_metrics = self.hparams.transducer_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("Transducer loss stats:\n")
                self.transducer_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "Transducer and PER stats written to file",
                    self.hparams.wer_file,
                )


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from timit_prepare import prepare_timit  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_timit(
        data_folder=hparams["data_folder"],
        splits=["train", "dev", "test"],
        save_folder=hparams["data_folder"],
    )

    # Collect index to label conversion dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["phn"]["index2lab"]

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"](), min_key="PER")
