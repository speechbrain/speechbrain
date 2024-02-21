#!/usr/bin/env python3
"""
Model for cascade and/or end-to-end spoken Dialogue State Tracking (DST).
This model uses a T5 backbone model on a spoken DST dataset.

Author
    * Lucas Druart 2024
"""

# import here because order of imports causes segfault
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import itertools
import speechbrain as sb

import logging

logger = logging.getLogger(__name__)


# Defining our Dialogue model
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

        semantics_enc_out = self.modules.semantic_encoder(
            semantics_tokens, semantics_lens
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

        decoder_out = self.modules.decoder(
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
        state = self.tokenizer.decode(hyp)
        with open(
            os.path.join(
                self.hparams.output_folder, "last_turns", f"{dialog_id}.txt",
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

        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        (loss / self.hparams.gradient_accumulation).requires_grad_().backward()

        if should_step:
            if not self.hparams.audio_frozen:
                self.audio_optimizer.step()
            if not self.hparams.semantic_encoder_frozen:
                self.semantic_optimizer.step()
            self.decoder_optimizer.step()
            self.fusion_optimizer.step()

            if not self.hparams.audio_frozen:
                self.audio_optimizer.zero_grad()
            if not self.hparams.semantic_encoder_frozen:
                self.semantic_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.fusion_optimizer.zero_grad()
            self.optimizer_step += 1

            # Update all the learning rates
            if not self.hparams.audio_frozen:
                old_audio_lr, new_audio_lr = self.hparams.lr_annealing_audio(
                    self.audio_optimizer
                )
            if not self.hparams.semantic_encoder_frozen:
                (
                    old_semantic_lr,
                    new_semantic_lr,
                ) = self.hparams.lr_annealing_semantics(self.semantic_optimizer)
            self.hparams.lr_annealing_fusion(self.fusion_optimizer)
            old_decoder_lr, new_decoder_lr = self.hparams.lr_annealing_decoder(
                self.decoder_optimizer
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
            if not self.hparams.semantic_encoder_frozen:
                self.hparams.train_logger.writer.add_scalar(
                    tag="Semantic Encoder LR",
                    scalar_value=self.semantic_optimizer.param_groups[0]["lr"],
                    global_step=self.optimizer_step,
                )
                self.hparams.text_logger.log_stats(
                    stats_meta={"Step": self.optimizer_step},
                    train_stats={
                        "Semantic Encoder LR": self.semantic_optimizer.param_groups[
                            0
                        ][
                            "lr"
                        ]
                    },
                )
            self.hparams.train_logger.writer.add_scalar(
                tag="Decoder LR",
                scalar_value=self.decoder_optimizer.param_groups[0]["lr"],
                global_step=self.optimizer_step,
            )
            self.hparams.text_logger.log_stats(
                stats_meta={"Step": self.optimizer_step},
                train_stats={
                    "Decoder LR": self.decoder_optimizer.param_groups[0]["lr"]
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

        if not self.hparams.semantic_encoder_frozen:
            self.semantic_optimizer = self.hparams.semantic_opt_class(
                self.modules.semantic_encoder.parameters()
            )

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

        self.decoder_optimizer = self.hparams.decoder_opt_class(
            self.modules.decoder.parameters()
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
        Overriding the write_previous_pred method to match the id format from SpokenWoz.
        """
        # Id in the form /path/to/dialogue_Turn-N
        dialog_id = element_id.split("/")[-1].split("_")[0]
        state = self.tokenizer.decode(hyp)
        with open(
            os.path.join(
                self.hparams.output_folder, "last_turns", f"{dialog_id}.txt",
            ),
            "w",
        ) as last_turn:
            last_turn.write(state + "\n")
