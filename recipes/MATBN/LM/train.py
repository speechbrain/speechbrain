import sys

import torch
import speechbrain as sb
from speechbrain.dataio import dataset
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml


class LM(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        tokens_bos, _ = batch.tokens_bos
        logits = self.hparams.model(tokens_bos)
        pred = self.hparams.log_softmax(logits)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        (loss / self.hparams.accumulation_steps).backward()

        if self.step % self.hparams.accumulation_steps == 0:
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams.lr_annealing,
                sb.nnet.schedulers.CyclicCosineScheduler,
            ):
                self.hparams.lr_annealing(self.optimizer)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            if not (
                isinstance(
                    self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
                )
                or isinstance(
                    self.hparams.lr_annealing,
                    sb.nnet.schedulers.CyclicCosineScheduler,
                )
            ):
                old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            else:
                old_lr = self.hparams.lr_annealing.current_lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["loss"],
            )

        if stage == sb.Stage.TEST and sb.utils.distributed.if_main_process():
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def dataio_prepare(hparams):
    @sb.utils.data_pipeline.takes("transcription")
    @sb.utils.data_pipeline.provides(
        "transcription", "tokens_bos", "tokens_eos"
    )
    def transcription_pipline(transcription):
        yield transcription
        tokens_list = hparams["tokenizer"].encode_as_ids(transcription)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    data_folder = hparams["data_folder"]
    datasets = {}
    for dataset_name in ["train", "dev", "test"]:
        json_path = f"{data_folder}/{dataset_name}.json"
        datasets[dataset_name] = dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[transcription_pipline],
            output_keys=["transcription", "tokens_bos", "tokens_eos"],
        )

    return datasets


if __name__ == "__main__":
    hparams_file_path, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file_path) as hparams_file:
        hparams = load_hyperpyyaml(hparams_file, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file_path,
        overrides=overrides,
    )

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    datasets = dataio_prepare(hparams)

    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # evaluation
    lm_brain.evaluate(
        datasets["test"],
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
