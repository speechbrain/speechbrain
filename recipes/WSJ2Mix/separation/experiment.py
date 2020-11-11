#!/usr/bin/python

"""
Recipe to train DP-RNN model on the WSJ0-2Mix dataset

Author:
    * Cem Subakan 2020
    * Mirko Bronzi 2020
"""

import argparse
import logging
import os
import pprint
import shutil
from pathlib import PosixPath

# import itertools as it

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import speechbrain as sb
from recipes.minimal_examples.neural_networks.separation.example_conv_tasnet import (
    create_minimal_data,
)
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
from speechbrain.utils.checkpoints import ckpt_recency

# from speechbrain.utils.train_logger import summarize_average

from speechbrain.data_io.data_io import write_wav_soundfile

# import speechbrain.nnet.lr_schedulers as schedulers
from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)


class SourceSeparationBrain(sb.core.Brain):
    def __init__(self, params, device, **kwargs):
        self.params = params
        self.device = device
        super(sb.core.Brain, self).__init__(**kwargs)
        self.eval_scores = []
        self.scaler = GradScaler()
        self.inifinite_loss_found = 0

    def compute_forward(self, mixture, stage="train", init_params=False):
        raise NotImplementedError("use a subclass")

    def compute_objectives(self, predictions, targets):
        if self.params.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        # train_onthefly option enables data augmentation,
        # by creating random mixtures within the batch
        if self.params.train_onthefly:
            bs = batch[0][1].shape[0]
            perm = torch.randperm(bs)

            T = 24000
            Tmax = max((batch[0][1].shape[-1] - T) // 10, 1)
            Ts = torch.randint(0, Tmax, (1,))
            source1 = batch[1][1][perm, Ts : Ts + T].to(self.device)
            source2 = batch[2][1][:, Ts : Ts + T].to(self.device)

            ws = torch.ones(2).to(self.device)
            ws = ws / ws.sum()

            inputs = ws[0] * source1 + ws[1] * source2
            targets = torch.cat(
                [source1.unsqueeze(1), source2.unsqueeze(1)], dim=1
            )
        else:
            inputs = batch[0][1].to(self.device)
            targets = torch.cat(
                [
                    batch[i][1].unsqueeze(-1)
                    for i in range(1, self.params.MaskNet.num_spks + 1)
                ],
                dim=-1,
            ).to(self.device)

        if isinstance(
            self.params.MaskNet.dual_mdl[0].intra_mdl,
            sb.lobes.models.dual_pathrnn.DPTNetBlock,
        ) or isinstance(
            self.params.MaskNet.dual_mdl[0].intra_mdl,
            sb.lobes.models.dual_pathrnn.PTRNNBlock,
        ):
            randstart = np.random.randint(
                0, 1 + max(0, inputs.shape[1] - 32000)
            )
            targets = targets[:, randstart : randstart + 32000, :]

        if self.params.use_data_augmentation:
            targets = targets.permute(0, 2, 1)
            targets = targets.reshape(-1, targets.shape[-1])
            wav_lens = torch.tensor([targets.shape[-1]] * targets.shape[0]).to(
                self.device
            )

            targets = self.params.augmentation(targets, wav_lens)
            targets = targets.reshape(
                -1, self.params.MaskNet.num_spks, targets.shape[-1]
            )
            targets = targets.permute(0, 2, 1)

            if hasattr(self.params, "use_data_shuffling"):
                # only would work for 2 spks
                perm = torch.randperm(targets.size(0))
                targets = torch.stack(
                    [targets[perm, :, 0], targets[:, :, 1]], dim=2
                )

            inputs = targets.sum(-1)

        # TODO: consider if we need this part..
        # if isinstance(self.params.lr_scheduler, schedulers.NoamScheduler):
        #     old_lr, new_lr = self.params.lr_scheduler(
        #         [self.optimizer], None, None
        #     )
        #     print("oldlr ", old_lr, "newlr", new_lr)
        #     print(self.optimizer.optim.param_groups[0]["lr"])

        if self.params.mixed_precision:
            with autocast():
                predictions = self.compute_forward(inputs)
                loss = self.compute_objectives(predictions, targets)

            if loss < 999999:  # fix for computational problems
                self.scaler.scale(loss).backward()
                if self.params.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.params.clip_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.inifinite_loss_found += 1
                logger.info(
                    "infinite loss! it happened {} times so far - skipping this batch".format(
                        self.inifinite_loss_found
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions = self.compute_forward(inputs)
            loss = self.compute_objectives(predictions, targets)
            loss.backward()
            if self.params.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.params.clip_grad_norm
                )
            self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs = batch[0][1].to(self.device)
        targets = torch.cat(
            [
                batch[i][1].unsqueeze(-1)
                for i in range(1, self.params.MaskNet.num_spks + 1)
            ],
            dim=-1,
        ).to(self.device)

        predictions = self.compute_forward(inputs, stage="test")
        loss = self.compute_objectives(predictions, targets)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):

        av_valid_loss = summarize_average(valid_stats["loss"])
        if isinstance(
            self.params.lr_scheduler, schedulers.ReduceLROnPlateau
        ) or isinstance(self.params.lr_scheduler, schedulers.DPRNNScheduler):
            current_lr, next_lr = self.params.lr_scheduler(
                [self.params.optimizer], epoch, av_valid_loss
            )
        else:
            # if we do not use the reducelronplateau, we do not change the lr
            next_lr = current_lr = self.params.optimizer.optim.param_groups[0][
                "lr"
            ]

        epoch_stats = {"epoch": epoch, "lr": current_lr}
        self.params.train_logger.log_stats(
            epoch_stats, train_stats, valid_stats
        )

        # TODO: find the new implementation for this metric/loss
        av_train_loss = summarize_average(train_stats["loss"])

        logger.info("Completed epoch %d" % epoch)
        logger.info(
            "Train SI-SNR: %.3f" % -summarize_average(train_stats["loss"])
        )
        eval_score = summarize_average(valid_stats["loss"])
        self.eval_scores.append(eval_score)
        logger.info("Valid SI-SNR: %.3f" % -eval_score)
        logger.info(
            "Current LR {} New LR on next epoch {}".format(current_lr, next_lr)
        )

        # TODO: check how it works in SB / and check where does this ckpt_recency comes from.
        self.params.checkpointer.save_and_keep_only(
            meta={"av_loss": av_valid_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["av_loss"]],
        )

    # TODO: consider if it's better or not to use the asme signature as the parent.
    def compute_forward(self, mixture, stage="train", init_params=False):
        """

        :param mixture: raw audio - dimension [batch_size, time]
        :param stage:
        :param init_params:
        :return:
        """

        mixture_w = self.params.Encoder(mixture, init_params=init_params)
        # [batch, channel, time / kernel stride]
        est_mask = self.params.MaskNet(mixture_w, init_params=init_params)

        out = [
            est_mask[i] * mixture_w for i in range(self.params.MaskNet.num_spks)
        ]
        est_source = torch.cat(
            [
                self.params.Decoder(out[i]).unsqueeze(-1)
                for i in range(self.params.MaskNet.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        # [B, T, Number of speaker=2]
        return est_source


# # Define training procedure
# class ASR(sb.Brain):
#     def compute_forward(self, x, y, stage):
#         """Forward computations from the waveform batches to the output probabilities."""
#         ids, wavs, wav_lens = x
#         ids, target_words, target_word_lens = y
#         wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
#
#         # Add augmentation if specified
#         if stage == sb.Stage.TRAIN:
#             if hasattr(self.modules, "env_corrupt"):
#                 wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
#                 wavs = torch.cat([wavs, wavs_noise], dim=0)
#                 wav_lens = torch.cat([wav_lens, wav_lens])
#                 target_words = torch.cat([target_words, target_words], dim=0)
#                 target_word_lens = torch.cat(
#                     [target_word_lens, target_word_lens]
#                 )
#             if hasattr(self.hparams, "augmentation"):
#                 wavs = self.hparams.augmentation(wavs, wav_lens)
#
#         # Prepare labels
#         target_tokens, _ = self.hparams.tokenizer(
#             target_words, target_word_lens, self.hparams.ind2lab, task="encode"
#         )
#         target_tokens = target_tokens.to(self.device)
#         y_in = sb.data_io.prepend_bos_token(
#             target_tokens, self.hparams.bos_index
#         )
#
#         # Forward pass
#         feats = self.hparams.compute_features(wavs)
#         feats = self.modules.normalize(feats, wav_lens)
#         x = self.modules.enc(feats.detach())
#         e_in = self.modules.emb(y_in)
#         h, _ = self.modules.dec(e_in, x, wav_lens)
#
#         # Output layer for seq2seq log-probabilities
#         logits = self.modules.seq_lin(h)
#         p_seq = self.hparams.log_softmax(logits)
#
#         # Compute outputs
#         if stage == sb.Stage.TRAIN:
#             current_epoch = self.hparams.epoch_counter.current
#             if current_epoch <= self.hparams.number_of_ctc_epochs:
#                 # Output layer for ctc log-probabilities
#                 logits = self.modules.ctc_lin(x)
#                 p_ctc = self.hparams.log_softmax(logits)
#                 return p_ctc, p_seq, wav_lens
#             else:
#                 return p_seq, wav_lens
#         else:
#             p_tokens, scores = self.hparams.beam_searcher(x, wav_lens)
#             return p_seq, wav_lens, p_tokens
#
#     def compute_objectives(self, predictions, targets, stage):
#         """Computes the loss (CTC+NLL) given predictions and targets."""
#
#         current_epoch = self.hparams.epoch_counter.current
#         if stage == sb.Stage.TRAIN:
#             if current_epoch <= self.hparams.number_of_ctc_epochs:
#                 p_ctc, p_seq, wav_lens = predictions
#             else:
#                 p_seq, wav_lens = predictions
#         else:
#             p_seq, wav_lens, predicted_tokens = predictions
#
#         ids, target_words, target_word_lens = targets
#         target_tokens, target_token_lens = self.hparams.tokenizer(
#             target_words, target_word_lens, self.hparams.ind2lab, task="encode"
#         )
#         target_tokens = target_tokens.to(self.device)
#         target_token_lens = target_token_lens.to(self.device)
#         if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
#             target_tokens = torch.cat([target_tokens, target_tokens], dim=0)
#             target_token_lens = torch.cat(
#                 [target_token_lens, target_token_lens], dim=0
#             )
#
#         # Add char_lens by one for eos token
#         abs_length = torch.round(target_token_lens * target_tokens.shape[1])
#
#         # Append eos token at the end of the label sequences
#         target_tokens_with_eos = sb.data_io.append_eos_token(
#             target_tokens, length=abs_length, eos_index=self.hparams.eos_index
#         )
#
#         # Convert to speechbrain-style relative length
#         rel_length = (abs_length + 1) / target_tokens_with_eos.shape[1]
#         loss_seq = self.hparams.seq_cost(
#             p_seq, target_tokens_with_eos, length=rel_length
#         )
#
#         # Add ctc loss if necessary
#         if (
#             stage == sb.Stage.TRAIN
#             and current_epoch <= self.hparams.number_of_ctc_epochs
#         ):
#             loss_ctc = self.hparams.ctc_cost(
#                 p_ctc, target_tokens, wav_lens, target_token_lens
#             )
#             loss = self.hparams.ctc_weight * loss_ctc
#             loss += (1 - self.hparams.ctc_weight) * loss_seq
#         else:
#             loss = loss_seq
#
#         if stage != sb.Stage.TRAIN:
#             # Decode token terms to words
#             predicted_words = self.hparams.tokenizer(
#                 predicted_tokens, task="decode_from_list"
#             )
#
#             # Convert indices to words
#             target_words = undo_padding(target_words, target_word_lens)
#             target_words = sb.data_io.convert_index_to_lab(
#                 target_words, self.hparams.ind2lab
#             )
#
#             self.wer_metric.append(ids, predicted_words, target_words)
#             self.cer_metric.append(ids, predicted_words, target_words)
#
#         return loss
#
#     def fit_batch(self, batch):
#         """Train the parameters given a single batch in input"""
#         inputs, targets = batch
#         predictions = self.compute_forward(inputs, targets, sb.Stage.TRAIN)
#         loss = self.compute_objectives(predictions, targets, sb.Stage.TRAIN)
#         loss.backward()
#         self.optimizer.step()
#         self.optimizer.zero_grad()
#         return loss.detach()
#
#     def evaluate_batch(self, batch, stage):
#         """Computations needed for validation/test batches"""
#         inputs, targets = batch
#         predictions = self.compute_forward(inputs, targets, stage=stage)
#         loss = self.compute_objectives(predictions, targets, stage=stage)
#         return loss.detach()
#
#     def on_stage_start(self, stage, epoch):
#         """Gets called at the beginning of each epoch"""
#         if stage != sb.Stage.TRAIN:
#             self.cer_metric = self.hparams.cer_computer()
#             self.wer_metric = self.hparams.error_rate_computer()
#
#     def on_stage_end(self, stage, stage_loss, epoch):
#         """Gets called at the end of a epoch."""
#         # Compute/store important stats
#         stage_stats = {"loss": stage_loss}
#         if stage == sb.Stage.TRAIN:
#             self.train_stats = stage_stats
#         else:
#             stage_stats["CER"] = self.cer_metric.summarize("error_rate")
#             stage_stats["WER"] = self.wer_metric.summarize("error_rate")
#
#         # Perform end-of-iteration things, like annealing, logging, etc.
#         if stage == sb.Stage.VALID:
#             old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
#             sb.nnet.update_learning_rate(self.optimizer, new_lr)
#
#             if self.root_process:
#                 self.hparams.train_logger.log_stats(
#                     stats_meta={"epoch": epoch, "lr": old_lr},
#                     train_stats=self.train_stats,
#                     valid_stats=stage_stats,
#                 )
#                 self.checkpointer.save_and_keep_only(
#                     meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
#                 )
#         elif stage == sb.Stage.TEST:
#             self.hparams.train_logger.log_stats(
#                 stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
#                 test_stats=stage_stats,
#             )
#             with open(self.hparams.wer_file, "w") as w:
#                 self.wer_metric.write_stats(w)
#
#     def load_tokenizer(self):
#         """Loads the sentence piece tokinizer specified in the yaml file"""
#         save_model_path = self.hparams.save_folder + "/tok_unigram.model"
#         save_vocab_path = self.hparams.save_folder + "/tok_unigram.vocab"
#
#         if hasattr(self.hparams, "tok_mdl_file"):
#             download_file(
#                 source=self.hparams.tok_mdl_file,
#                 dest=save_model_path,
#                 replace_existing=True,
#             )
#             self.hparams.tokenizer.sp.load(save_model_path)
#
#         if hasattr(self.hparams, "tok_voc_file"):
#             download_file(
#                 source=self.hparams.tok_voc_file,
#                 dest=save_vocab_path,
#                 replace_existing=True,
#             )
#
#     def load_lm(self):
#         """Loads the LM specified in the yaml file"""
#         save_model_path = os.path.join(
#             self.hparams.output_folder, "save", "lm_model.ckpt"
#         )
#         download_file(self.hparams.lm_ckpt_file, save_model_path)
#
#         # Load downloaded model, removing prefix
#         state_dict = torch.load(save_model_path)
#         state_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items()}
#         self.hparams.lm_model.load_state_dict(state_dict, strict=True)
#         self.hparams.lm_model.eval()
#
#


# TODO: complete the main / check the one above from the updated code.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file", required=True)
    parser.add_argument(
        "--data_path", help="the data path to load the dataset", required=False
    )
    parser.add_argument(
        "--minimal",
        help="will run a minimal example for debugging",
        action="store_true",
    )
    parser.add_argument(
        "--test_only",
        help="will only run testing, and not training",
        action="store_true",
    )
    parser.add_argument(
        "--use_multigpu",
        help="will use multigpu in training",
        action="store_true",
    )
    parser.add_argument(
        "--num_spks", help="number of speakers", type=int, default=2,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.minimal:
        repo_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../"
        params = create_minimal_data(repo_path, args.config)
        logger.info("setting epoch size to 1 - because --minimal")
        params["N_epochs"] = 1
        # params = fix_params_for_orion(params)
    else:
        with open(args.config) as fin:
            params = sb.yaml.load_extended_yaml(fin)

        # override the data_path if we want to
        if args.data_path is not None:
            params.wsj0mixpath = args.data_path

        # this points to the folder to which we will save the wsj0-mix dataset
        data_save_dir = params.wsj0mixpath

        # if the dataset is not present, we create the dataset
        if not os.path.exists(data_save_dir):
            from recipes.WSJ2Mix.prepare_data import get_wsj_files

            # this points to the folder which holds the wsj0 dataset folder
            wsj0path = params.wsj0path
            get_wsj_files(wsj0path, data_save_dir)

        # load or create the csv files which enables us to get the speechbrain dataloaders
        # if not (
        #    os.path.exists(params.save_folder + "/wsj_tr.csv")
        #    and os.path.exists(params.save_folder + "/wsj_cv.csv")
        #    and os.path.exists(params.save_folder + "/wsj_tt.csv")
        # ):
        # we always recreate the csv files too keep track of the latest path

        if params.MaskNet.num_spks == 2:
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv

            create_wsj_csv(data_save_dir, params.save_folder)
        elif params.MaskNet.num_spks == 3:
            from recipes.WSJ2Mix.prepare_data import create_wsj_csv_3spks

            create_wsj_csv_3spks(data_save_dir, params.save_folder)
        else:
            raise ValueError("We do not support this many speakers")

        tr_csv = os.path.realpath(
            os.path.join(params.save_folder + "/wsj_tr.csv")
        )
        cv_csv = os.path.realpath(
            os.path.join(params.save_folder + "/wsj_cv.csv")
        )
        tt_csv = os.path.realpath(
            os.path.join(params.save_folder + "/wsj_tt.csv")
        )

        with open(args.config) as fin:
            params = sb.yaml.load_extended_yaml(
                fin, {"tr_csv": tr_csv, "cv_csv": cv_csv, "tt_csv": tt_csv}
            )
        params = fix_params_for_orion(params)
        # copy the config file for book keeping
        shutil.copyfile(args.config, params.output_folder + "/config.txt")

    logger.info(pprint.PrettyPrinter(indent=4).pformat(params))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "will run on device {} / using mixed precision? {}".format(
            device, params["mixed_precision"]
        )
    )

    train_loader = params["train_loader"]()
    val_loader = params["val_loader"]()
    test_loader = params["test_loader"]()

    ctn = SourceSeparationBrain(
        modules=[
            params["Encoder"],  # .to(device),
            params["MaskNet"],  # .to(device),
            params["Decoder"],  # .to(device),
        ],
        optimizer=params["optimizer"],
        first_inputs=[next(iter(train_loader))[0][1]],
        params=params,
        device=device,
    )
    ctn.args = args

    for module in ctn.modules:
        reset_layer_recursively(module)

    if args.use_multigpu and torch.cuda.device_count() > 1:
        # ctn.modules[i] = torch.nn.DataParallel(ctn.modules[i]).to(device)
        print("will train on multiple gpus")
        ctn.params.Encoder = torch.nn.DataParallel(ctn.params["Encoder"]).to(
            device
        )
        ctn.params.MaskNet = torch.nn.DataParallel(ctn.params["MaskNet"]).to(
            device
        )
        ctn.params.Decoder = torch.nn.DataParallel(ctn.params["Decoder"]).to(
            device
        )
    else:
        print("will train on single gpu")
        ctn.params.Encoder = ctn.params["Encoder"].to(device)
        ctn.params.MaskNet = ctn.params["MaskNet"].to(device)
        ctn.params.Decoder = ctn.params["Decoder"].to(device)

    params.checkpointer.recover_if_possible(lambda c: -c.meta["av_loss"])

    if args.test_only:
        save_audio_results(params, ctn, test_loader, device, N=10)

        # get the score on the whole test set
        test_stats = ctn.evaluate(test_loader)
        logger.info(
            "Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"])
        )
    else:
        # mlflow.start_run()
        ctn.fit(
            range(params["N_epochs"]),
            train_set=train_loader,
            valid_set=val_loader,
            progressbar=params["progressbar"],
            early_stopping_with_patience=params["early_stopping_with_patience"],
        )
        mlflow.end_run()

        test_stats = ctn.evaluate(test_loader)
        logger.info(
            "Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"])
        )

        best_eval = min(ctn.eval_scores)
        logger.info("Best result on validation: {}".format(-best_eval))

        report_results(
            [dict(name="dev_metric", type="objective", value=float(best_eval),)]
        )


# if __name__ == "__main__":
#     # This hack needed to import data preparation script from ../..
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
#     from librispeech_prepare import prepare_librispeech  # noqa E402
#
#     # Load hyperparameters file with command-line overrides
#     hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
#     with open(hparams_file) as fin:
#         hparams = sb.load_extended_yaml(fin, overrides)
#
#     # Create experiment directory
#     sb.create_experiment_directory(
#         experiment_directory=hparams["output_folder"],
#         hyperparams_to_save=hparams_file,
#         overrides=overrides,
#     )
#
#     # Prepare data
#     prepare_librispeech(
#         data_folder=hparams["data_folder"],
#         splits=hparams["train_splits"]
#         + [hparams["dev_split"], "test-clean", "test-other"],
#         merge_lst=hparams["train_splits"],
#         merge_name=hparams["csv_train"],
#         save_folder=hparams["data_folder"],
#     )
#
#     # Creating tokenizer must be done after preparation
#     # Specify the bos_id/eos_id if different from blank_id
#     hparams["tokenizer"] = SentencePiece(
#         model_dir=hparams["save_folder"],
#         vocab_size=hparams["output_neurons"],
#         csv_train=hparams["csv_train"],
#         csv_read="wrd",
#         model_type=hparams["token_type"],
#         character_coverage=1.0,
#     )
#
#     train_set = hparams["train_loader"]()
#     valid_set = hparams["valid_loader"]()
#     test_clean_set = hparams["test_clean_loader"]()
#     test_other_set = hparams["test_other_loader"]()
#     hparams["ind2lab"] = hparams["test_other_loader"].label_dict["wrd"][
#         "index2lab"
#     ]
#
#     # Brain class initialization
#     asr_brain = ASR(
#         modules=hparams["modules"],
#         opt_class=hparams["opt_class"],
#         hparams=hparams,
#         checkpointer=hparams["checkpointer"],
#         device=hparams["device"],
#         multigpu_count=hparams["multigpu_count"],
#         multigpu_backend=hparams["multigpu_backend"],
#     )
#
#     asr_brain.load_tokenizer()
#     if hasattr(asr_brain.hparams, "lm_ckpt_file"):
#         asr_brain.load_lm()
#
#     # Training
#     asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
#
#     # Test
#     asr_brain.hparams.wer_file = (
#         hparams["output_folder"] + "/wer_test_clean.txt"
#     )
#     asr_brain.evaluate(test_clean_set)
#     asr_brain.hparams.wer_file = (
#         hparams["output_folder"] + "/wer_test_other.txt"
#     )
#     asr_brain.evaluate(test_other_set)
