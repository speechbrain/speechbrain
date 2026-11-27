#!/usr/bin/env python3
"""Recipe for fine-tuning SeamlessM4T for the Tamasheq-French ST task
(no transcriptions). The system employs the speech encoder and the text
decoder of SeamlessM4T from Meta (https://arxiv.org/abs/2312.05187).

To run this recipe, do the following:
> python train_with_seamless.py hparams/train_seamless_st.yaml --root_data_folder=your/data/path

Author
------
 * Othman Istaiteh 2026
"""

import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio
        tokens_bos, _ = batch.tokens_bos  # translation

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # Forward encoder + decoder
        enc_out, enc_lens, logits = self.modules.seamless(
            wavs, tokens_bos, wav_lens
        )
        p_seq = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), enc_lens
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), enc_lens)

        return p_seq, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_seq, hyps) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        if stage != sb.Stage.TRAIN:
            detokenized_translation = [
                self.detokenizer.detokenize(translation.split(" "))
                for translation in batch.trans
            ]
            # it needs to be a list of list due to the extend on the bleu implementation
            targets = [detokenized_translation]

            predictions = [
                self.detokenizer.detokenize(hyp.split(" "))
                for hyp in self.tokenizer.batch_decode(
                    hyps, skip_special_tokens=True
                )
            ]

            self.bleu_metric.append(ids, predictions, targets)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.lr_annealing_seamless(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.detokenizer = MosesDetokenizer(lang=self.hparams.lang)

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
            stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing_seamless.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": lr,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"BLEU": stage_stats["BLEU"], "epoch": epoch},
                max_keys=["BLEU"],
                num_to_keep=self.hparams.num_checkpoints_to_keep,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.bleu_file, "w", encoding="utf-8") as w:
                    self.bleu_metric.write_stats(w)


def dataio_prepare(hparams, seamless):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    def load_dataset(json_path):
        """Loads a json manifest with a numeric duration, as
        prepare_iwslt22.py stores the durations as strings."""
        data = sb.dataio.dataio.load_data_json(
            json_path, replacements={"data_root": data_folder}
        )
        for item in data.values():
            item["duration"] = float(item["duration"])
        return sb.dataio.dataset.DynamicItemDataset(data)

    # 1. Define datasets
    train_data = load_dataset(hparams["annotation_train"])

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # We also sort the validation and test data so it is faster to evaluate
    valid_data = load_dataset(hparams["annotation_valid"]).filtered_sorted(
        sort_key="duration"
    )
    test_data = load_dataset(hparams["annotation_test"]).filtered_sorted(
        sort_key="duration"
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("trans")
    @sb.utils.data_pipeline.provides("trans", "tokens_bos", "tokens_eos")
    def text_pipeline(translation):
        """Processes the translations to generate proper labels"""
        translation = translation.replace("\n", "")
        yield translation
        # [target_lang, tokens..., eos]
        tokens_list = seamless.encode_target(translation)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "trans", "tokens_bos", "tokens_eos"],
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    import prepare_iwslt22

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_iwslt22.data_proc,
            kwargs={
                "dataset_folder": hparams["root_data_folder"],
                "output_folder": hparams["data_folder"],
            },
        )

    # Defining tokenizer and loading it
    tokenizer = hparams["seamless"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(
        hparams, hparams["seamless"]
    )

    # Trainer initialization
    st_brain = ST(
        modules=hparams["modules"],
        opt_class=hparams["seamless_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    st_brain.tokenizer = tokenizer

    # Training
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for name, dataset in [("valid", valid_data), ("test", test_data)]:
        st_brain.hparams.bleu_file = (
            hparams["output_folder"] + f"/bleu_{name}.txt"
        )
        st_brain.evaluate(
            dataset,
            max_key="BLEU",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
