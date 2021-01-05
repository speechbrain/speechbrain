#!/usr/bin/env python3
"""Recipe for doing ASR with phoneme targets and CTC loss on Voicebank

To run this recipe, do the following:
> python experiment.py {hyperparameter file} --data_folder /path/to/noisy-vctk

Use your own hyperparameter file or the provided `hyperparams.yaml`

To use noisy inputs, change `input_type` field from `clean_wav` to `noisy_wav`.
To use pretrained model, enter path in `pretrained` field.

Authors
 * Peter Plantinga 2020
"""
import sys
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs = self.modules.augmentation(wavs, wav_lens)
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        out = self.modules.model(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        pout, pout_lens = predictions
        phns, phn_lens = batch.phn_encoded
        loss = self.hparams.compute_cost(pout, phns, pout_lens, phn_lens)
        self.ctc_metrics.append(batch.id, pout, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                pout, pout_lens, blank_id=-1
            )
            self.per_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
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
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print("CTC and PER stats written to ", self.hparams.per_file)


def data_io_prep(hparams):
    """Creates the datasets and their data processing pipelines"""

    label_encoder = sb.data_io.encoder.CTCTextEncoder()

    # 1. Define audio pipeline:
    @sb.utils.data_pipeline.takes(hparams["input_type"])
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.data_io.data_io.read_audio(wav)
        return sig

    # 2. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    # 3. Create datasets
    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.data_io.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root", hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "phn_encoded"],
        )

    # Sort train dataset and ensure it doesn't get un-sorted
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        data["train"] = data["train"].filtered_sorted(
            sort_key="duration", reverse=hparams["sorting"] == "descending",
        )
        hparams["dataloader_options"]["train_shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    # 4. Fit encoder to train data
    label_encoder.insert_blank(index=hparams["blank_index"])
    label_encoder.update_from_didataset(data["train"], output_key="phn_list")

    return data, label_encoder


# Begin Recipe!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    # Initialize ddp (necessary for multi-GPU DDP training)
    sb.ddp_init_group(run_opts)

    # Prepare data on one process
    from voicebank_prepare import prepare_voicebank  # noqa E402

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
        },
    )

    data, label_encoder = data_io_prep(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Load pretrained model
    if "pretrained" in hparams:
        state_dict = torch.load(hparams["pretrained"])
        hparams["modules"]["model"].load_state_dict(state_dict)

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    asr_brain.fit(asr_brain.hparams.epoch_counter, data["train"], data["valid"])
    asr_brain.evaluate(data["test"], min_key="PER")
