#!/usr/bin/env python3
"""Recipe for doing ASR with phoneme targets and CTC loss on the TIMIT dataset

To run this recipe, do the following:
> python experiment.py hyperparams.yaml --data_folder /path/to/TIMIT

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import timit_prepare
import speechbrain as sb


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # Adding augmentation when specified:
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        pout, pout_lens = predictions
        ids = batch.id
        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)

        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(ids, pout, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=phns,
                target_len=phn_lens,
                lab2ind=self.hparams.label_encoder.decode_ndim,
            )

        return loss

    def on_stage_start(self, stage, epoch):
        self.ctc_metrics = self.hparams.ctc_stats()

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
                meta={"PER": per}, min_keys=["PER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print("CTC and PER stats written to ", self.hparams.wer_file)


def make_datasets(data_folder, batch_size):

    sb.utils.data_utils.run_on_main(
        timit_prepare.prepare_timit,
        data_folder=data_folder,
        splits=["train", "dev", "test"],
        save_folder=data_folder,
    )
    label_encoder = sb.data_io.encoder.CTCTextEncoder()

    def split_phn(phn):
        return {"phn_list": sb.utils.data_utils.split_by_whitespace(phn)}

    def make_phn_encoded(phn_list):
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        return {"phn_encoded": phn_encoded}

    def make_sig(wav):
        return {"sig": sb.data_io.data_io.read_audio(wav)}

    item_transforms = {
        "phn": split_phn,
        "phn_list": make_phn_encoded,
        "wav": make_sig,
    }

    train_data = sb.data_io.transform_dataset.TransformDataset.from_csv(
        csv_path=os.path.join(data_folder, "train.csv"),
        replacements={"data_root": data_folder},
        item_transforms=item_transforms,
        output_keys=["id", "phn_encoded", "sig"],
    )
    train_loader = sb.data_io.dataloader.SaveableDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sb.data_io.batch.PaddedBatch,
    )
    valid_data = sb.data_io.transform_dataset.TransformDataset.from_csv(
        csv_path=os.path.join(data_folder, "dev.csv"),
        replacements={"data_root": data_folder},
        item_transforms=item_transforms,
        output_keys=["id", "phn_encoded", "sig"],
    )
    valid_loader = sb.data_io.dataloader.SaveableDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sb.data_io.batch.PaddedBatch,
    )
    test_data = sb.data_io.transform_dataset.TransformDataset.from_csv(
        csv_path=os.path.join(data_folder, "test.csv"),
        replacements={"data_root": data_folder},
        item_transforms=item_transforms,
        output_keys=["id", "phn_encoded", "sig"],
    )
    test_loader = sb.data_io.dataloader.SaveableDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sb.data_io.batch.PaddedBatch,
    )

    # Create label encoding
    if not label_encoder.load_if_possible("encoder_state.txt"):
        label_encoder.update_from_transform_dataset(
            train_data, output_key="phn_list", sequence_input=True
        )
        label_encoder.update_from_transform_dataset(
            valid_data, output_key="phn_list", sequence_input=True
        )
        label_encoder.update_from_transform_dataset(
            test_data, output_key="phn_list", sequence_input=True
        )
        label_encoder.insert_blank(index=hparams["blank_index"])
        label_encoder.save("encoder_state.txt")

    return train_loader, valid_loader, test_loader, label_encoder


# Begin Recipe!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_loader, valid_loader, test_loader, label_encoder = make_datasets(
        hparams["data_folder"], hparams["batch_size"],
    )
    hparams["label_encoder"] = label_encoder

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_loader, valid_loader)
    asr_brain.evaluate(test_loader, min_key="PER")
