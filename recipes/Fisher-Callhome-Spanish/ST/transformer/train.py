#!/usr/bin/env/python3
"""Recipe for training a Transformer based ST system with Fisher-Callhome.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/conformer.yaml

Authors
 * YAO-FEI, CHENG 2021
"""

import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer

import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
en_detokenizer = MosesDetokenizer(lang="en")


class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch.sig

        tokens_bos, _ = batch.tokens_bos  # for translation task
        transcription_bos, _ = batch.transcription_bos  # for asr task
        transcription_tokens, _ = batch.transcription_tokens  # for mt task

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        asr_p_seq = None
        # asr output layer for seq2seq log-probabilities
        if self.hparams.asr_weight > 0 and self.hparams.ctc_weight < 1:
            asr_pred = self.modules.Transformer.forward_asr(
                enc_out,
                src,
                transcription_bos,
                wav_lens,
                pad_idx=self.hparams.pad_index,
            )
            asr_pred = self.modules.asr_seq_lin(asr_pred)
            asr_p_seq = self.hparams.log_softmax(asr_pred)

        # st output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # asr ctc
        p_ctc = None
        if self.hparams.ctc_weight > 0:
            logits = self.modules.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(logits)

        # mt task
        mt_p_seq = None
        if self.hparams.mt_weight > 0:
            _, mt_pred = self.modules.Transformer.forward_mt(
                transcription_tokens,
                tokens_bos,
                pad_idx=self.hparams.pad_index,
            )

            # mt output layer for seq2seq log-probabilities
            mt_pred = self.modules.seq_lin(mt_pred)
            mt_p_seq = self.hparams.log_softmax(mt_pred)

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

        return p_ctc, p_seq, asr_p_seq, mt_p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_ctc, p_seq, asr_p_seq, mt_p_seq, wav_lens, hyps) = predictions

        ids = batch.id

        tokens_eos, tokens_eos_lens = batch.tokens_eos
        transcription_eos, transcription_eos_lens = batch.transcription_eos
        transcription_tokens, transcription_lens = batch.transcription_tokens

        # loss for different tasks
        # asr loss = ctc_weight * ctc loss + (1 - ctc_weight) * asr attention loss
        # mt loss = mt attention loss
        # st loss =
        # (1 - asr_weight - mt_weight) * st attention loss +
        # asr_weight * asr loss +
        # mt_weight * mt loss
        attention_loss = 0
        asr_ctc_loss = 0
        asr_attention_loss = 0
        mt_loss = 0

        # st attention loss
        attention_loss = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # asr attention loss
        if self.hparams.ctc_weight < 1 and self.hparams.asr_weight > 0:
            asr_attention_loss = self.hparams.seq_cost(
                asr_p_seq, transcription_eos, length=transcription_eos_lens
            )

        # asr ctc loss
        if self.hparams.ctc_weight > 0 and self.hparams.asr_weight > 0:
            asr_ctc_loss = self.hparams.ctc_cost(
                p_ctc, transcription_tokens, wav_lens, transcription_lens
            )

        # mt attention loss
        if self.hparams.mt_weight > 0:
            mt_loss = self.hparams.seq_cost(
                mt_p_seq, tokens_eos, length=tokens_eos_lens
            )

        asr_loss = (self.hparams.ctc_weight * asr_ctc_loss) + (
            1 - self.hparams.ctc_weight
        ) * asr_attention_loss
        loss = (
            (1 - self.hparams.asr_weight - self.hparams.mt_weight)
            * attention_loss
            + self.hparams.asr_weight * asr_loss
            + self.hparams.mt_weight * mt_loss
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if stage == sb.Stage.TEST:
                # 4 references bleu score
                predictions = [
                    en_detokenizer.detokenize(
                        hparams["tokenizer"].decode_ids(utt_seq).split(" ")
                    )
                    for utt_seq in hyps
                ]

                four_references = [
                    batch.translation_0,
                    batch.translation_1,
                    batch.translation_2,
                    batch.translation_3,
                ]

                targets = []
                for reference in four_references:
                    detokenized_translation = [
                        en_detokenizer.detokenize(translation.split(" "))
                        for translation in reference
                    ]
                    targets.append(detokenized_translation)

                self.bleu_metric.append(ids, predictions, targets)
            elif (
                current_epoch % valid_search_interval == 0
                and stage == sb.Stage.VALID
            ):
                predictions = [
                    en_detokenizer.detokenize(
                        hparams["tokenizer"].decode_ids(utt_seq).split(" ")
                    )
                    for utt_seq in hyps
                ]

                targets = [
                    en_detokenizer.detokenize(translation.split(" "))
                    for translation in batch.translation_0
                ]
                self.bleu_metric.append(ids, predictions, [targets])

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def on_fit_batch_start(self, batch, should_step):
        """Gets called at the beginning of each fit_batch."""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
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
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if stage == sb.Stage.TEST:
                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
            elif (
                current_epoch % valid_search_interval == 0
                and stage == sb.Stage.VALID
            ):
                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            with open(self.hparams.bleu_file, "a+", encoding="utf-8") as w:
                self.bleu_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:
                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible()

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes("translation_0")
    @sb.utils.data_pipeline.provides(
        "translation_0", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def one_reference_text_pipeline(translation):
        """Processes the transcriptions to generate proper labels"""
        yield translation
        tokens_list = hparams["tokenizer"].encode_as_ids(translation)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    @sb.utils.data_pipeline.takes(
        "translation_0", "translation_1", "translation_2", "translation_3"
    )
    @sb.utils.data_pipeline.provides(
        "translation_0",
        "translation_1",
        "translation_2",
        "translation_3",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def four_reference_text_pipeline(*translations):
        """Processes the transcriptions to generate proper labels"""
        yield translations[0]
        yield translations[1]
        yield translations[2]
        yield translations[3]
        tokens_list = hparams["tokenizer"].encode_as_ids(translations[0])
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
        "transcription_list",
        "transcription_bos",
        "transcription_eos",
        "transcription_tokens",
    )
    def transcription_text_pipeline(transcription):
        yield transcription
        tokens_list = hparams["tokenizer"].encode_as_ids(transcription)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train", "dev"]:
        json_path = f"{data_folder}/{dataset}/data.json"
        dataset = dataset if dataset == "train" else "valid"

        is_use_sp = dataset == "train" and "speed_perturb" in hparams
        audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[
                audio_pipeline_func,
                one_reference_text_pipeline,
                transcription_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "translation_0",
                "tokens_bos",
                "tokens_eos",
                "tokens",
                "transcription",
                "transcription_list",
                "transcription_bos",
                "transcription_eos",
                "transcription_tokens",
            ],
        )

    for dataset in ["dev", "dev2", "test"]:
        json_path = f"{data_folder}/{dataset}/data.json"
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[
                audio_pipeline,
                four_reference_text_pipeline,
                transcription_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "duration",
                "translation_0",
                "translation_1",
                "translation_2",
                "translation_3",
                "tokens_bos",
                "tokens_eos",
                "tokens",
                "transcription",
                "transcription_list",
                "transcription_bos",
                "transcription_eos",
                "transcription_tokens",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration"
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration"
            )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration", reverse=True
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration", reverse=True
            )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 3},
                key_max_value={"duration": 5},
                sort_key="duration",
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
            )

        hparams["train_dataloader_opts"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # transcription/translation tokenizer
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    st_brain = ST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    for dataset in ["dev", "dev2", "test"]:
        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
