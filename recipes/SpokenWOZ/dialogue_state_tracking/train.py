#!/usr/bin/env python3
"""
Recipe to fine-tune and infer for cascade and/or end-to-end spoken Dialogue State Tracking (DST).
This model fine-tunes a T5 backbone model on a spoken DST dataset.

To run this recipe, do the following:
> python train.py hparams/train_spokenwoz[_with_whisper_enc].yaml --data_folder YOUR_DATA_FOLDER

Author
    * Lucas Druart 2024
"""

import itertools
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoTokenizer

import speechbrain as sb
from speechbrain.augment.time_domain import Resample
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class DialogueUnderstanding(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """
        Forward computation of the model involves a forward pass on both
        the audio and semantic encoders and a forward pass on the decoder.
        """

        batch = batch.to(self.device)
        wavs, wavs_lens = batch.sig
        semantics_tokens, semantics_lens = batch.semantics_tokens
        outputs_tokens, outputs_lens = batch.outputs_tokens

        # Replacing the speechbrain padding tokens (id 0) with T5 tokenizer's padding id
        # Careful since the bos token in output_tokens also has the id 0
        semantics_tokens = torch.where(
            semantics_tokens == 0, self.tokenizer.pad_token_id, semantics_tokens
        )
        outputs_tokens_nobos = torch.where(
            outputs_tokens == 0, self.tokenizer.pad_token_id, outputs_tokens
        )[:, 1:]
        outputs_tokens = torch.cat(
            (
                torch.zeros_like(outputs_tokens[:, 0]).unsqueeze(-1),
                outputs_tokens_nobos,
            ),
            axis=1,
        ).to(outputs_tokens.device)

        # Add augmentation if specified
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.version == "e2e"
            and hasattr(self.hparams, "augmentation")
        ):
            wavs, wavs_lens = self.hparams.augmentation(wavs, wavs_lens)

        semantics_enc_out = self.modules.textual_model.forward_encoder(
            semantics_tokens
        )

        if "cascade" in self.hparams.version:
            encoder_out = semantics_enc_out
        elif self.hparams.version == "e2e":
            audio_enc_out = self.modules.audio_encoder(wavs)
            if self.hparams.downsampling:
                audio_enc_out = self.modules.conv1(audio_enc_out)
                audio_enc_out = self.hparams.conv_activation(audio_enc_out)
                audio_enc_out = self.hparams.dropout(audio_enc_out)
                audio_enc_out = self.modules.conv2(audio_enc_out)
                audio_enc_out = self.hparams.conv_activation(audio_enc_out)
                audio_enc_out = self.hparams.dropout(audio_enc_out)
            enc_concat = torch.cat((semantics_enc_out, audio_enc_out), dim=-2)
            encoder_out, _ = self.modules.fusion(enc_concat)
        else:
            raise KeyError(
                'hparams attribute "version" should be one of "cascade[_model]" or "e2e".'
            )

        decoder_out = self.modules.textual_model.forward_decoder(
            encoder_hidden_states=encoder_out, decoder_input_ids=outputs_tokens
        )

        logprobs = self.hparams.log_softmax(decoder_out)

        hyps = None

        if self.step % self.hparams.debug_print == 0:
            # To not slow down training only looking at selected tokens with teacher forcing
            hyps = torch.argmax(logprobs, dim=-1)

        if stage != sb.Stage.TRAIN:
            with torch.no_grad():
                # Searcher returns ids minus more than one token
                hyps, _, _, _ = self.hparams.valid_greedy_search(
                    encoder_out.detach(), wavs_lens
                )

        if stage == sb.Stage.VALID:
            target_tokens, target_lens = batch.outputs_tokens_nobos
            if not os.path.isdir(self.hparams.pred_folder):
                os.mkdir(self.hparams.pred_folder)
            with open(
                os.path.join(self.hparams.pred_folder, f"dev_{self.epoch}.csv"),
                "a",
                encoding="utf-8",
            ) as pred_file:
                for hyp, element_id in zip(hyps, batch.id):
                    pred_file.write(
                        f"{element_id},{self.tokenizer.decode(hyp)}\n"
                    )

            # Updating running accuracy
            self.accuracy.append(
                [self.tokenizer.decode(hyp) for hyp in hyps],
                [
                    self.tokenizer.decode(targ, skip_special_tokens=True)
                    for targ in target_tokens
                ],
            )

        elif stage == sb.Stage.TEST:
            # Writing the predictions in a file for future evaluation
            with open(self.hparams.output_file, "a") as pred_file:
                for hyp, element_id in zip(hyps, batch.id):
                    pred_file.write(
                        f"{element_id},{self.tokenizer.decode(hyp)}\n"
                    )

                    # Keeping track of the last predicted state of each dialogue to use it for the next prediction
                    if not self.hparams.gold_previous_state:
                        self.write_previous_pred(element_id, hyp)

        return logprobs, hyps, wavs_lens

    def write_previous_pred(self, element_id, hyp):

        # Id in the form /path/to/dialogue/Turn-N
        dialog_id = element_id.split("/")[-2]
        state = self.tokenizer.decode(hyp).replace("[State] ", "")
        with open(
            os.path.join(
                self.hparams.output_folder,
                "last_turns",
                f"{dialog_id}.txt",
            ),
            "w",
        ) as last_turn:
            last_turn.write(state + "\n")

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes and returns the loss.
        """
        logprobs, hyps, wavs_lens = predictions
        batch = batch.to(self.device)
        outputs_tokens, outputs_lens = batch.outputs_tokens_nobos

        # Replacing the speechbrain padding tokens (id 0) with T5 tokenizer's padding id
        outputs_tokens = torch.where(
            outputs_tokens == 0, self.tokenizer.pad_token_id, outputs_tokens
        ).to(outputs_tokens.device)

        loss = self.hparams.nll_loss(
            logprobs, outputs_tokens, length=outputs_lens
        )

        return loss

    def fit_batch(self, batch):
        """
        Performs a forward and backward pass on a batch.
        """
        should_step = self.step % self.hparams.gradient_accumulation == 0
        debug_step = self.step % self.hparams.debug_print == 0

        # with fp16, the loss explodes almost immediately for this model.
        # could reinvestigate in the future and rework `fit_batch`.
        assert self.precision == "fp32", "AMP is not supported for this model"

        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        (loss / self.hparams.gradient_accumulation).requires_grad_().backward()

        if should_step:
            if not self.hparams.audio_frozen:
                self.audio_optimizer.step()
            self.text_optimizer.step()
            self.fusion_optimizer.step()

            if not self.hparams.audio_frozen:
                self.audio_optimizer.zero_grad()
            self.text_optimizer.zero_grad()
            self.fusion_optimizer.zero_grad()
            self.optimizer_step += 1

            # Update all the learning rates
            if not self.hparams.audio_frozen:
                old_audio_lr, new_audio_lr = self.hparams.lr_annealing_audio(
                    self.audio_optimizer
                )
            self.hparams.lr_annealing_fusion(self.fusion_optimizer)
            old_decoder_lr, new_decoder_lr = self.hparams.lr_annealing_text(
                self.text_optimizer
            )

            # Logging the loss and lrs
            self.hparams.train_logger.writer.add_scalar(
                tag="Loss",
                scalar_value=loss.detach().cpu(),
                global_step=self.optimizer_step,
            )
            self.hparams.text_logger.log_stats(
                stats_meta={"Step": self.optimizer_step},
                train_stats={"Loss: ": loss.detach().cpu()},
            )

            self.hparams.train_logger.writer.add_scalar(
                tag="Fusion LR",
                scalar_value=self.fusion_optimizer.param_groups[0]["lr"],
                global_step=self.optimizer_step,
            )
            self.hparams.text_logger.log_stats(
                stats_meta={"Step": self.optimizer_step},
                train_stats={
                    "Fusion LR": self.fusion_optimizer.param_groups[0]["lr"]
                },
            )

            if not self.hparams.audio_frozen:
                self.hparams.train_logger.writer.add_scalar(
                    tag="Audio Encoder LR",
                    scalar_value=self.audio_optimizer.param_groups[0]["lr"],
                    global_step=self.optimizer_step,
                )
                self.hparams.text_logger.log_stats(
                    stats_meta={"Step": self.optimizer_step},
                    train_stats={
                        "Audio Encoder LR": self.audio_optimizer.param_groups[
                            0
                        ]["lr"]
                    },
                )
            self.hparams.train_logger.writer.add_scalar(
                tag="Text LR",
                scalar_value=self.text_optimizer.param_groups[0]["lr"],
                global_step=self.optimizer_step,
            )
            self.hparams.text_logger.log_stats(
                stats_meta={"Step": self.optimizer_step},
                train_stats={
                    "Text LR": self.text_optimizer.param_groups[0]["lr"]
                },
            )

        # Log the predictions and expected outputs for debug
        if debug_step:
            outputs_tokens, _ = batch.outputs_tokens_nobos
            semantics_tokens, _ = batch.semantics_tokens
            previous_states = self.tokenizer.batch_decode(
                semantics_tokens, skip_special_tokens=True
            )
            predictions = self.tokenizer.batch_decode(
                outputs[1], skip_special_tokens=False
            )
            targets = self.tokenizer.batch_decode(
                outputs_tokens, skip_special_tokens=True
            )

            log_text = "  \n  \n".join(
                [
                    f"Semantic input: {semantic}  \nPredicted output: {prediction}  \nExpected output: {target}\n"
                    for semantic, prediction, target in zip(
                        previous_states, predictions, targets
                    )
                ]
            )
            self.hparams.train_logger.writer.add_text(
                tag="DEBUG-Train",
                text_string=log_text,
                global_step=self.optimizer_step,
            )
            try:
                self.hparams.text_logger.log_stats(
                    stats_meta={"Step": self.optimizer_step},
                    train_stats={"DEBUG-Train": log_text},
                )
            except UnicodeEncodeError:
                pass

        return loss.detach().cpu()

    def init_optimizers(self):
        """
        Initializing the three different optimizers.
        """
        if not self.hparams.audio_frozen:
            if self.hparams.freeze_feature_extractor:
                audio_params = [
                    param
                    for name, param in self.modules.audio_encoder.named_parameters()
                    if "model.feature_extractor" not in name
                    and "model.feature_projection" not in name
                ]
            else:
                audio_params = self.modules.audio_encoder.parameters()
            self.audio_optimizer = self.hparams.audio_opt_class(audio_params)

        fusion_params = self.modules.fusion.parameters()
        if self.hparams.downsampling:
            fusion_params = itertools.chain.from_iterable(
                [
                    fusion_params,
                    self.modules.conv1.parameters(),
                    self.modules.conv2.parameters(),
                ]
            )
        self.fusion_optimizer = self.hparams.fusion_opt_class(fusion_params)

        self.text_optimizer = self.hparams.text_opt_class(
            self.modules.textual_model.parameters()
        )

    def on_stage_start(self, stage, epoch):
        self.stage = stage
        if stage == sb.Stage.VALID:
            self.accuracy = self.hparams.acc_computer()
            if not os.path.isdir(self.hparams.pred_folder):
                os.mkdir(self.hparams.pred_folder)
            # Emptying file
            with open(
                os.path.join(self.hparams.pred_folder, f"dev_{self.epoch}.csv"),
                "w",
            ):
                pass

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.epoch = epoch
        elif stage == sb.Stage.VALID:
            stage_stats["Accuracy"] = self.accuracy.summarize()
            self.hparams.train_logger.writer.add_scalar(
                tag="Dev Accuracy",
                scalar_value=stage_stats["Accuracy"],
                global_step=self.optimizer_step,
            )
            self.hparams.text_logger.log_stats(
                stats_meta={"Step": self.optimizer_step},
                train_stats={"Dev Accuracy": stage_stats["Accuracy"]},
            )

            self.checkpointer.save_checkpoint()

    def on_evaluate_start(self, max_key=None, min_key=None):
        # Opening and closing the file to reset it
        with open(self.hparams.output_file, "w"):
            pass
        super().on_evaluate_start(max_key, min_key)


class SpokenWozUnderstanding(DialogueUnderstanding):
    def write_previous_pred(self, element_id, hyp):
        """
        Overriding the write_previous_pred method to match the id format from SpokenWOZ.
        """
        # Id in the form /path/to/dialogue_Turn-N
        dialog_id = element_id.split("/")[-1].split("_")[0]
        state = self.tokenizer.decode(hyp)
        with open(
            os.path.join(
                self.hparams.output_folder,
                "last_turns",
                f"{dialog_id}.txt",
            ),
            "w",
        ) as last_turn:
            last_turn.write(state + "\n")


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
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
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]}
        )

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_datasets = {}
    for csv_file in hparams["valid_csv"]:
        name = Path(csv_file).stem
        valid_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        valid_datasets[name] = valid_datasets[name].filtered_sorted(
            sort_key="duration"
        )
    valid_data = valid_datasets[Path(hparams["valid_csv"][0]).stem]

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem

        if hparams["sorting_turns"]:
            # Sorting by turn nbr to always have the previous dialogue state already processed,
            # default is ascending
            ordered_csv = csv_file.replace(".csv", "_sorted.csv")
            df = pd.read_csv(csv_file)
            df.sort_values(by="turnID", inplace=True)
            df.to_csv(ordered_csv, header=True, index=False)
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=ordered_csv, replacements={"data_root": data_folder}
            )
        else:
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_file, replacements={"data_root": data_folder}
            )
            test_datasets[name] = test_datasets[name].filtered_sorted(
                sort_key="duration"
            )
        hparams["valid_loader_kwargs"]["shuffle"] = False

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("id", "wav", "start", "end")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(id, wav, start, end):
        resampler = Resample(orig_freq=8000, new_freq=16000)
        sig = read_audio(wav)
        sig = sig[int(start) : int(end)]
        sig = sig.unsqueeze(0)  # Must be B*T*C
        resampled = resampler(sig)
        # Fusing both channels
        resampled = torch.mean(resampled, dim=2)
        # Selecting the correct frames: start*2 bc resampled
        sig = torch.squeeze(resampled)

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes(
        "id", "previous_state", "agent", "user", "current_state"
    )
    @sb.utils.data_pipeline.provides(
        "semantics",
        "semantics_tokens",
        "outputs",
        "outputs_tokens",
        "outputs_tokens_nobos",
    )
    def text_pipeline(ID, previous_state, agent, user, current_state):

        if hparams["gold_previous_state"]:
            state = previous_state
        else:
            dialogue_id = ID.split("/")[-1].split("_")[0]
            turn_id = int(ID.split("/")[-1].split("_")[1].replace("Turn-", ""))
            if turn_id > 1:
                assert os.path.isfile(
                    os.path.join(
                        hparams["output_folder"],
                        "last_turns",
                        f"{dialogue_id}.txt",
                    )
                )
                with open(
                    os.path.join(
                        hparams["output_folder"],
                        "last_turns",
                        f"{dialogue_id}.txt",
                    ),
                    "r",
                ) as last_turn:
                    for line in last_turn:
                        dialogue_last_turn = line.strip()
                state = dialogue_last_turn
            else:
                state = ""

        if "cascade" in hparams["version"]:
            semantics = f"[State] {state} [Agent] {agent} [User] {user}"
        elif hparams["version"] == "e2e":
            semantics = f"[State] {state}"
        else:
            raise KeyError(
                'hparams attribute "version" should be set to "cascade_model" or "e2e".'
            )
        yield semantics

        semantics_tokens = tokenizer.encode(semantics)
        semantics_tokens = torch.LongTensor(semantics_tokens)
        yield semantics_tokens

        outputs = current_state
        yield outputs

        # T5 uses the pad_token_id as bos token
        tokens_list = [tokenizer.pad_token_id]
        tokens_list += tokenizer.encode(f"[State] {current_state}")

        yield torch.LongTensor(tokens_list[:-1])

        yield torch.LongTensor(tokens_list[1:])

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "semantics_tokens",
            "sig",
            "outputs_tokens",
            "outputs_tokens_nobos",
        ],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not hparams["skip_prep"]:

        # Dataset preparation
        from spokenwoz_prepare import prepare_spokenwoz

        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_spokenwoz,
            kwargs={
                "data_folder": hparams["data_folder"],
                "version": hparams["version"],
                "tr_splits": hparams["train_splits"],
                "dev_splits": hparams["dev_splits"],
                "te_splits": hparams["test_splits"],
                "save_folder": hparams["save_folder"],
                "merge_lst": hparams["train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams["skip_prep"],
                "select_n_sentences": hparams["select_n_sentences"],
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(hparams["t5_hub"])

    if hparams["gold_previous_state"]:
        hparams["sorting_turns"] = False
    else:
        hparams["sorting_turns"] = True

    # Datasets creation (tokenization)
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    slu_brain = SpokenWozUnderstanding(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    slu_brain.tokenizer = tokenizer

    if not hparams["inference"]:

        slu_brain.fit(
            slu_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_loader_kwargs"],
            valid_loader_kwargs=hparams["valid_loader_kwargs"],
        )

    else:
        # Testing
        for k in test_datasets.keys():
            if not hparams["gold_previous_state"]:
                # Storing the last dialog's turn prediction
                if not os.path.isdir(
                    os.path.join(hparams["output_folder"], "last_turns")
                ):
                    os.mkdir(
                        os.path.join(hparams["output_folder"], "last_turns")
                    )
                slu_brain.hparams.output_file = os.path.join(
                    hparams["pred_folder"], "{}_previous.csv".format(k)
                )
            else:
                slu_brain.hparams.output_file = os.path.join(
                    hparams["pred_folder"], "{}.csv".format(k)
                )
            if not os.path.isfile(slu_brain.hparams.output_file):
                slu_brain.evaluate(
                    test_datasets[k],
                    test_loader_kwargs=hparams["valid_loader_kwargs"],
                )
