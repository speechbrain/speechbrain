import sys

import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml


class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wavs_len = batch.sig
        tokens_bos, _ = batch.tokens_bos
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wavs_len, epoch=current_epoch)

        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wavs_len, pad_idx=self.hparams.pad_index
        )

        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wavs_len)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wavs_len)

        return p_ctc, p_seq, wavs_len, hyps

    def compute_objectives(self, predictions, batch, stage):

        (p_ctc, p_seq, wavs_len, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_len = batch.tokens_eos
        tokens, tokens_len = batch.tokens

        attention_loss = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_len
        )
        ctc_loss = self.hparams.ctc_cost(p_ctc, tokens, wavs_len, tokens_len)
        loss = (
            self.hparams.ctc_weight * ctc_loss
            + (1 - self.hparams.ctc_weight) * attention_loss
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                predictions = [
                    hparams["tokenizer"].decode_ids(utt_seq).split(" ")
                    for utt_seq in hyps
                ]
                targets = [
                    transcription.split(" ")
                    for transcription in batch.transcription
                ]
                if self.hparams.remove_spaces:
                    predictions = [
                        "".join(prediction_words)
                        for prediction_words in predictions
                    ]
                    targets = [
                        "".join(target_words) for target_words in targets
                    ]
                    self.cer_metric.append(ids, predictions, targets)

            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_len)

        return loss

    def fit_batch(self, batch):
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        # origin function is call loss.detach().cpu()
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

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
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as cer_file:
                self.cer_metric.write_stats(cer_file)

            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
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

        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return
                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start()

        checkpointers = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        checkpointer = sb.utils.checkpoints.average_checkpoints(
            checkpointers, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(checkpointer, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    @sb.utils.data_pipeline.takes("transcription")
    @sb.utils.data_pipeline.provides(
        "transcription", "tokens_bos", "tokens_eos", "tokens"
    )
    def transcription_pipline(transcription):
        yield transcription
        tokens_list = hparams["tokenizer"].encode_as_ids(transcription)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    datasets = {}
    data_folder = hparams["data_folder"]
    output_keys = [
        "transcription",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "sig",
        "id",
    ]
    default_dynamic_items = [transcription_pipline, audio_pipline]
    train_dynamic_item = [transcription_pipline, sp_audio_pipline]

    for dataset_name in ["train", "dev", "test"]:
        if dataset_name == "train":
            dynamic_items = train_dynamic_item
        else:
            dynamic_items = default_dynamic_items

        json_path = f"{data_folder}/{dataset_name}.json"
        datasets[dataset_name] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        )

    return datasets


if __name__ == "__main__":
    hparams_file_path, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file_path) as hparams_file:
        hparams = load_hyperpyyaml(hparams_file, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file_path,
        overrides=overrides,
    )

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    datasets = dataio_prepare(hparams)

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # asr_brain.evaluate(
    #     datasets["test"],max_key="ACC", test_loader_kwargs=hparams["test_dataloader_opts"]
    # )
