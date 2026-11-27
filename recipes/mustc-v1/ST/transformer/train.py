#!/usr/bin/env/python3
"""Recipe for training a Transformer based ST system with MuST-C version 1.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.
To run this recipe, do the following:
> python train.py hparams/transformer.yaml
Authors
 * YAO-FEI, CHENG 2021
"""

import sys
import torch
import logging

import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch.sig

        if hasattr(self.hparams, "speed_perturb"):
            wavs = self.hparams.speed_perturb(wavs)

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

        # compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, asr_p_seq, mt_p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_ctc, p_seq, asr_p_seq, mt_p_seq, wav_lens, hyps,) = predictions

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
            p_seq, tokens_eos, length=tokens_eos_lens,
        )

        # asr attention loss
        if self.hparams.ctc_weight < 1 and self.hparams.asr_weight > 0:
            asr_attention_loss = self.hparams.seq_cost(
                asr_p_seq, transcription_eos, length=transcription_eos_lens,
            )

            # asr ctc loss
        if self.hparams.ctc_weight > 0 and self.hparams.asr_weight > 0:
            asr_ctc_loss = self.hparams.ctc_cost(
                p_ctc, transcription_tokens, wav_lens, transcription_lens,
            )

        # mt attention loss
        if self.hparams.mt_weight > 0:
            mt_loss = self.hparams.seq_cost(
                mt_p_seq, tokens_eos, length=tokens_eos_lens,
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

            is_valid_epoch = current_epoch % valid_search_interval == 0
            is_valid_search = is_valid_epoch and stage == sb.Stage.VALID
            if stage == sb.Stage.TEST or is_valid_search:
                predicted_words = [
                    hparams["tokenizer"].decode_ids(utt_seq) for utt_seq in hyps
                ]
                prediction = ["".join(words) for words in predicted_words]
                targets = [translation for translation in batch.translation]

                self.bleu_metric.append(ids, prediction, [targets])

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

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
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
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
                num_to_keep=5,
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

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        audio = {"file": wav, "start": start, "stop": stop}
        sig = sb.dataio.dataio.read_audio(audio)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes("translation")
    @sb.utils.data_pipeline.provides(
        "translation", "tokens_list", "tokens_bos", "tokens_eos", "tokens",
    )
    def translation_text_pipeline(translation):
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
    target_language = hparams["target_language"]
    for dataset in ["train", "dev", "test_he", "test_com"]:
        json_path = f"{data_folder}/{dataset}_en-{target_language}.json"
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[
                audio_pipeline,
                translation_text_pipeline,
                transcription_text_pipeline,
            ],
            output_keys=[
                "id",
                "sig",
                "translation",
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
        datasets[dataset] = datasets[dataset].filtered_sorted(
            key_max_value={"duration": 30}, sort_key="duration",
        )

    for dataset in ["train", "valid", "test"]:
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = True

    if hparams["debug"]:
        datasets["train"] = datasets["train"].filtered_sorted(
            key_min_value={"duration": 3},
            key_max_value={"duration": 5},
            sort_key="duration",
        )
        datasets["dev"] = datasets["dev"].filtered_sorted(
            key_min_value={"duration": 3}, key_max_value={"duration": 5},
        )

        hparams["train_dataloader_opts"]["shuffle"] = True

    train_batch_sampler = None
    dev_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.batch import PaddedBatch  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]

        hop_size = dynamic_hparams["feats_hop_size"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            datasets["train"],
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            seed=hparams["seed"],
        )

        dev_batch_sampler = DynamicBatchSampler(
            datasets["dev"],
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            seed=hparams["seed"],
        )

    return datasets, train_batch_sampler, dev_batch_sampler


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # transcription/translation tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # We can now directly create the datasets for training, dev, and test
    datasets, train_batch_sampler, dev_batch_sampler = dataio_prepare(hparams)

    train_dataloader_opts = hparams["train_dataloader_opts"]
    dev_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_batch_sampler is not None:
        train_dataloader_opts = {"batch_sampler": train_batch_sampler}
    if dev_batch_sampler is not None:
        dev_dataloader_opts = {"batch_sampler": dev_batch_sampler}

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
        datasets["dev"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=dev_dataloader_opts,
    )

    for dataset in ["test_com", "test_he"]:
        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
