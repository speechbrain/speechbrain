#!/usr/bin/env/python3
"""

AISHELL-1 NAR-BERT-ASR training recipe.
https://arxiv.org/pdf/2104.04805.pdf

"""

import sys
import torch
from torch.nn.utils.rnn import pad_sequence
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from prepare import prepare_aishell
from laso import PositionalEncoding

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        pos_encodings, _ = batch.pos_encodings

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "SpeedPerturb"):
                wavs_speed_90 = self.hparams.SpeedPerturb["speed_90"](wavs)
                wavs_speed_100 = wavs
                wavs_speed_110 = self.hparams.SpeedPerturb["speed_110"](wavs)

                wav_lens = torch.cat(
                    [
                        wav_lens
                        * (wavs_speed_90.size(1) / wavs_speed_110.size(1)),
                        wav_lens
                        * (wavs_speed_100.size(1) / wavs_speed_110.size(1)),
                        wav_lens,
                    ]
                )
                wavs = pad_sequence(
                    [*wavs_speed_90, *wavs_speed_100, *wavs_speed_110],
                    batch_first=True,
                )
                pos_encodings = torch.cat(
                    [pos_encodings, pos_encodings, pos_encodings], dim=0
                )
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                pos_encodings = torch.cat([pos_encodings, pos_encodings], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "SpecAugment"):
                feats = self.hparams.SpecAugment(feats)

        # forward modules
        laso_embeds = self.hparams.LASO.encode(
            src=feats,
            tgt=pos_encodings,
            wav_len=wav_lens,
            pad_idx=self.hparams.pad_index,
        )
        pred = self.hparams.BERT(inputs_embeds=laso_embeds).logits
        p_seq = self.hparams.log_softmax(pred)

        return p_seq

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        ids = batch.id
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "SpeedPerturb"):
                tokens = torch.cat([tokens, tokens, tokens], dim=0)
                tokens_lens = torch.cat(
                    [tokens_lens, tokens_lens, tokens_lens], dim=0
                )
            if hasattr(self.modules, "env_corrupt"):
                tokens = torch.cat([tokens, tokens], dim=0)
                tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            log_probabilities=predictions, targets=tokens, length=tokens_lens
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                predicted_words_list = []
                target_words_list = [[c for c in seq] for seq in batch.wrd]

                for prediction in predictions:
                    # Decode token terms to words
                    predicted_tokens = self.tokenizer.convert_ids_to_tokens(
                        torch.argmax(prediction, dim=-1).cpu().numpy()
                    )

                    predicted_words = []
                    for c in predicted_tokens:
                        if c == "[CLS]":
                            continue
                        elif c == "[SEP]" or c == "[PAD]":
                            break
                        else:
                            predicted_words.append(c)

                    predicted_words_list.append(predicted_words)

                self.cer_metric.append(
                    ids=ids,
                    predict=predicted_words_list,
                    target=target_words_list,
                )

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(
                log_probabilities=predictions,
                targets=tokens,
                length=tokens_lens,
            )
        return loss_seq

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

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

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
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages acccording current stage
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
                meta={"CER": stage_stats["CER"], "epoch": epoch},
                min_keys=["CER"],
                num_to_keep=self.hparams.keep_nbest_models,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evalation stage
            # delete the rest of the intermediate checkpoints
            # CER is set to 0 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"CER": 0, "epoch": epoch},
                min_keys=["CER"],
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
        """Initilaize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitilaize the optimizer
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
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={"data_root": data_folder},
    )

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

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # LASO-PDS positional encoding
    positional_encoding = PositionalEncoding(hparams["d_model"])

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("wrd", "tokens", "pos_encodings")
    def text_pipeline(wrd):
        yield wrd

        tokens_list = tokenizer(wrd)["input_ids"]
        if len(tokens_list) > hparams["maximum_length"]:
            tokens_list = tokens_list[: hparams["maximum_length"]]
        elif len(tokens_list) < hparams["maximum_length"]:
            tokens_list = tokens_list + [hparams["pad_index"]] * (
                hparams["maximum_length"] - len(tokens_list)
            )

        tokens = torch.LongTensor(tokens_list)
        yield tokens

        pos_encodings = positional_encoding(tokens.unsqueeze(0)).squeeze(0)
        yield pos_encodings

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens", "pos_encodings"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing AISHELL-1)
    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)

    # load pretrain weights
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )
