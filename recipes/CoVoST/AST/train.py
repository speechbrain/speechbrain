#!/usr/bin/env python3
"""Recipe for training a Transformer AST system with CoVoST
The system employs an encoder, a decoder, and an attention mechanism
between them. An additional CTC loss can be used for warmup with an ASR task.

To run this recipe, do the following:
> python train.py hparams/conformer_large.yaml

Author
------
 * Titouan Parcollet 2025
"""

import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class AST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)

        # Add feature augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "fea_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            feats, _ = self.hparams.fea_augment(feats, wav_lens)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc ASR log-probabilities if warming up
        if self.optimizer_step < self.hparams.asr_warmup_steps:
            logits = self.modules.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(logits)
        else:
            p_ctc = None

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if is_valid_search:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )

        elif is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, predicted_tokens) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens_asr, tokens_asr_lens = batch.tokens_asr

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # ASR warmup with CTC if specified.
        if self.optimizer_step < self.hparams.asr_warmup_steps:
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens_asr, wav_lens, tokens_asr_lens
            )
            loss = (
                self.hparams.ctc_weight * loss_ctc
                + (1 - self.hparams.ctc_weight) * loss_seq
            )
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                predictions = [
                    self.tgt_detokenizer.detokenize(
                        self.tokenizer.sp.decode_ids(utt_seq).split(" ")
                    )
                    for utt_seq in predicted_tokens
                ]

                detokenized_translation = [
                    self.tgt_detokenizer.detokenize(translation.split(" "))
                    for translation in batch.translation
                ]

                # it needs to be a list of list due to the extend on the bleu implementation
                targets = [detokenized_translation]

                self.bleu_metric.append(ids, predictions, targets)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

        self.tgt_detokenizer = MosesDetokenizer(lang=self.hparams.tgt_language)

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
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
                stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()
            stage_stats["ACC"] = self.acc_metric.summarize()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


# Define custom data procedure
def dataio_prepare(hparams, tokenizer_ast, tokenizer_asr):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    train_data = train_data.filtered_sorted(
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_val_test"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_val_test"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define AST and ASR text pipelines:
    @sb.utils.data_pipeline.takes("translation")
    @sb.utils.data_pipeline.provides(
        "translation",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def st_text_pipeline(translation):
        yield translation
        tokens_list = tokenizer_ast.sp.encode_as_ids(translation)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    @sb.utils.data_pipeline.takes("transcription")
    @sb.utils.data_pipeline.provides(
        "transcription",
        "tokens_asr_list",
        "tokens_asr_bos",
        "tokens_asr_eos",
        "tokens_asr",
    )
    def asr_text_pipeline(transcription):
        yield transcription
        tokens_asr_list = tokenizer_asr.sp.encode_as_ids(transcription)
        yield tokens_asr_list
        tokens_asr_bos = torch.LongTensor(
            [hparams["bos_index"]] + (tokens_asr_list)
        )
        yield tokens_asr_bos
        tokens_asr_eos = torch.LongTensor(
            tokens_asr_list + [hparams["eos_index"]]
        )
        yield tokens_asr_eos
        tokens_asr = torch.LongTensor(tokens_asr_list)
        yield tokens_asr

    sb.dataio.dataset.add_dynamic_item(datasets, st_text_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, asr_text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "translation",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "tokens_asr_bos",
            "tokens_asr_eos",
            "tokens_asr",
        ],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_data,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
    from covost_prepare import prepare_covost  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_covost,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "src_language": hparams["src_language"],
            "tgt_language": hparams["tgt_language"],
            "skip_prep": hparams["skip_prep"],
            "convert_to_wav": hparams["convert_to_wav"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer_ast = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="translation",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
        text_file="target",
    )

    tokenizer_asr = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["asr_output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="transcription",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
        text_file="source",
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer_ast, tokenizer_asr)

    # Trainer initialization
    ast_brain = AST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    ast_brain.tokenizer = tokenizer_ast

    # Manage dynamic batching
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # Training
    ast_brain.fit(
        ast_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing

    ast_brain.evaluate(
        valid_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    ast_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
