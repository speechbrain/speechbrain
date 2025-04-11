#!/usr/bin/python3
"""This recipe to train L-MAC to interpret audio classifiers.

The command to run for this recipe, with WHAM augmentation (as used in the LMAC-TD paper)
    python train_lmactd.py hparams/lmactd_cnn14.yaml --data_folder=/yourpath/ESC50 --add_wham_noise True --wham_folder=/yourpath/wham_noise

For more details, please refer to the README file.


Authors
    * Eleonora Mancini 2025
    * Francesco Paissan 2025
    * Cem Subakan 2025
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
from esc50_prepare import dataio_prep, prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from interpreter_brain import InterpreterBrain
from wham_prepare import combine_batches, prepare_wham

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.distributed import run_on_main

#from pdb import set_trace as bp


eps = 1e-10


def tv_loss(mask, tv_weight=1, power=2, border_penalty=0.3):
    if tv_weight is None or tv_weight == 0:
        return 0.0
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/PiotrDabkowski/pytorch-saliency/blob/bfd501ec7888dbb3727494d06c71449df1530196/sal/utils/mask.py#L5
    w_variance = torch.sum(torch.pow(mask[:, :, :-1] - mask[:, :, 1:], power))
    h_variance = torch.sum(torch.pow(mask[:, :-1, :] - mask[:, 1:, :], power))

    loss = tv_weight * (h_variance + w_variance) / float(power * mask.size(0))
    return loss


class LMACTD(InterpreterBrain):

    def crosscor(self, spectrogram, template):
        """Compute the cross correlation metric defined in the L-MAC paper, used in finetuning"""
        if self.hparams.crosscortype == "conv":
            spectrogram = spectrogram - spectrogram.mean((-1, -2), keepdim=True)
            template = template - template.mean((-1, -2), keepdim=True)
            template = template.unsqueeze(1)
            # 1 x BS x T x F
            # BS x 1 x T x F
            tmp = F.conv2d(
                spectrogram[None],
                template,
                bias=None,
                groups=spectrogram.shape[0],
            )

            normalization1 = F.conv2d(
                spectrogram[None] ** 2,
                torch.ones_like(template),
                groups=spectrogram.shape[0],
            )
            normalization2 = F.conv2d(
                torch.ones_like(spectrogram[None]),
                template**2,
                groups=spectrogram.shape[0],
            )

            ncc = (
                tmp / torch.sqrt(normalization1 * normalization2 + 1e-8)
            ).squeeze()

            return ncc
        elif self.hparams.crosscortype == "dotp":
            dotp = (spectrogram * template).mean((-1, -2))
            norms_specs = spectrogram.pow(2).mean((-1, -2)).sqrt()
            norms_templates = template.pow(2).mean((-1, -2)).sqrt()
            norm_dotp = dotp / (norms_specs * norms_templates)
            return norm_dotp
        else:
            raise ValueError("unknown crosscor type!")

    
    def interpret_computation_steps(self, wavs, print_probability=False):
        """Computation steps to get the interpretation spectrogram"""
        _, X_mel, _, _ = self.preprocess(wavs)

        # Embeddings + sound classifier
        hcat, _, _, _ = self.classifier_forward(X_mel)

        wavs_encoded = self.modules.encoder(wavs)
        decoder_out = self.modules.convt_decoder(hcat)
        decoder_out = decoder_out.squeeze(1).permute(0, 2, 1)[
            :, :, : wavs_encoded.shape[-1]
        ]

        al = self.hparams.alpha_mix
        est_mask = self.modules.masknet(
            wavs_encoded * (1 - al) + decoder_out * (al)
        ).squeeze(0)

        sep_h = wavs_encoded * est_mask
        # the [0] is bc we only have one source
        interp = self.modules.decoder(sep_h)
        mask_out_t = self.modules.decoder((wavs_encoded * (1 - est_mask)))

        # Preliminary operations for visualization
        X_stft_logpower_interp, X_mel_interp, _, X_stft_power_interp = (
            self.preprocess(interp)
        )

        return X_stft_logpower_interp.transpose(1, 2), interp

    def compute_forward(self, batch, stage):
        """Forward computation defined for to generate the saliency maps with L-MAC"""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # augment batch with WHAM!
        if hasattr(self.hparams, "add_wham_noise"):
            if self.hparams.add_wham_noise:
                wavs = combine_batches(wavs, iter(self.hparams.wham_dataset))
        X_stft_logpower, X_mel, _, X_stft_power = self.preprocess(wavs)

        # Embeddings + sound classifier
        hcat, _, predictions, _ = self.classifier_forward(X_mel)

        wavs_encoded = self.modules.encoder(wavs)
        decoder_out = self.modules.convt_decoder(hcat)
        decoder_out = decoder_out.squeeze(1).permute(0, 2, 1)[
            :, :, : wavs_encoded.shape[-1]
        ]

        al = self.hparams.alpha_mix
        est_mask = self.modules.masknet(
            wavs_encoded * (1 - al) + decoder_out * (al)
        ).squeeze(0)

        sep_h = wavs_encoded * est_mask
        # the [0] is bc we only have one source
        interp = self.modules.decoder(sep_h)
        mask_out_t = self.modules.decoder((wavs_encoded * (1 - est_mask)))

        # Preliminary operations for visualization
        X_stft_logpower_interp, X_mel_interp, _, X_stft_power_interp = (
            self.preprocess(interp)
        )
        est_mask_time_domain = self.modules.decoder(est_mask)
        X_stft_logpower_mask, X_mel_mask, _, _ = self.preprocess(
            est_mask_time_domain
        )

        saliency_map = X_mel_interp / (
            X_mel + 1e-8
        )  # Add a small constant to avoid division by zero

        if stage == sb.Stage.VALID:
            # save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                self.viz_ints(
                    wavs,
                    X_stft_logpower,
                    interp,
                    X_stft_logpower_interp,
                    est_mask_time_domain,
                    X_stft_logpower_mask,
                    est_mask,
                    saliency_map,
                    batch,
                )

        if stage == sb.Stage.TEST and self.hparams.save_interpretations:
            # During TEST save always, if required
            self.viz_ints(
                wavs,
                X_stft_logpower,
                interp,
                X_stft_logpower_interp,
                est_mask_time_domain,
                X_stft_logpower_mask,
                est_mask,
                saliency_map,
                batch,
            )

        return ((wavs, lens), predictions, interp, hcat, est_mask, mask_out_t)

    def extra_metrics(self):
        """This function defines the extra metrics required for L-MAC.
        This is limited to the counter() function which is used to count the number of data items which passes the crosscorrelation threshold, during the finetuning stage of L-MAC.
        """

        def counter(c):
            return c

        # return {"in_masks": counter}
        return {}

    def compute_objectives(self, pred, batch, stage):
        """Helper function to compute the objectives"""
        (batch_sig, predictions, interp, _, est_mask, mask_out_t) = pred

        batch = batch.to(self.device)
        wavs_clean, _ = batch.sig

        # taking them from forward because they are augmented there!
        wavs, _ = batch_sig

        uttid = batch.id
        labels, _ = batch.class_string_encoded

        (
            X_stft_logpower_clean,
            _,
            _,
            _,
        ) = self.preprocess(wavs_clean)
        X_stft_logpower, _, _, _ = self.preprocess(wavs)

        # map clean to same dimensionality
        # X_stft_logpower_clean = X_stft_logpower_clean[:, :Tmax, :]

        X_stft_logpower_interp, X_interp_mel, _, _ = self.preprocess(interp)
        _, mask_out_mel, _, _ = self.preprocess(mask_out_t)

        # mask_in = xhat * X_stft_logpower[:, :Tmax, :]
        # mask_out = (1 - xhat) * X_stft_logpower[:, :Tmax, :]

        mask_in_preds = self.classifier_forward(X_interp_mel)[2]
        mask_out_preds = self.classifier_forward(mask_out_mel)[2]

        class_pred = predictions.argmax(1)
        l_in = F.nll_loss(mask_in_preds.log_softmax(1), class_pred)
        l_out = -F.nll_loss(mask_out_preds.log_softmax(1), class_pred)

        ao_loss = l_in * self.hparams.l_in_w + self.hparams.l_out_w * l_out

        rec_loss = (
            self.hparams.rec_loss(
                (mask_out_t + interp)[..., None], wavs[..., None]
            )
        ).mean() * self.hparams.reg_w_sum

        # l1_reg = est_mask.abs().mean() * self.hparams.reg_w_l1
        l1_reg = X_stft_logpower_interp.abs().mean() * self.hparams.reg_w_l1

        mask_in_preds = mask_in_preds.softmax(1)

        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.inp_fid.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AD.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AI.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AG.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.sps.append(uttid, wavs, X_stft_logpower, labels)
            self.comp.append(uttid, wavs, X_stft_logpower, labels)
            self.faithfulness.append(
                uttid,
                predictions.softmax(1),
                mask_out_preds.softmax(1),
            )

        # self.in_masks.append(uttid, c=crosscor_mask)
        self.acc_metric.append(
            uttid,
            predict=predictions,
            target=labels,
        )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return ao_loss + l1_reg + rec_loss


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    print("Eval only hparams:")
    print("overlap_type=", hparams["overlap_type"])
    print("int_method=", hparams["int_method"])
    print("ljspeech_path=", hparams["ljspeech_path"])
    print("single_sample=", hparams["single_sample"])

    print("Inherited hparams:")
    print("use_melspectra_log1p=", hparams["use_melspectra_log1p"])

    print(
        "Interpreter class is inheriting the train_logger",
        hparams["train_logger"],
    )

    # classifier is fixed here
    hparams["embedding_model"].eval()
    hparams["classifier"].eval()
    hparams["embedding_model"].requires_grad_(False)
    hparams["classifier"].requires_grad_(False)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    # create WHAM dataset according to hparams
    if "wham_folder" in hparams:
        hparams["wham_dataset"] = prepare_wham(
            hparams["wham_folder"],
            hparams["add_wham_noise"],
            hparams["sample_rate"],
            hparams["signal_length_s"],
            hparams["wham_audio_folder"],
        )

        assert hparams["signal_length_s"] == 5, "Fix wham sig length!"
        # assert hparams["out_n_neurons"] == 50, "Fix number of outputs classes!"

    class_labels = list(label_encoder.ind2lab.values())
    hparams["class_labels"] = class_labels
    print("Class Labels:", class_labels)

    if hparams["finetuning"]:
        if hparams["pretrained_interpreter"] is None:
            raise AssertionError(
                "You should specify pretrained model for finetuning."
            )

    Interpreter_brain = LMACTD(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["pretrained_interpreter"] is not None and hparams["finetuning"]:
        print("Load pretrained_interpreter for interpreer finetuning...")
        run_on_main(hparams["load_pretrained"].collect_files)
        hparams["load_pretrained"].load_collected()

    if "pretrained_esc50" in hparams and hparams["use_pretrained"]:
        print("Loading model...")
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    if hparams["pretrained_ed_path"] is not None:
        print(
            "Loading pretrained encoder-decoder for interpreter finetuning..."
        )
        run_on_main(hparams["load_encoder"].collect_files)
        hparams["load_encoder"].load_collected()
        if hparams["freeze_encoder"]:
            hparams["Encoder"].requires_grad_(False)
        if hparams["freeze_decoder"]:
            hparams["Decoder"].requires_grad_(False)

    hparams["embedding_model"].to(Interpreter_brain.device)
    hparams["classifier"].to(Interpreter_brain.device)
    hparams["embedding_model"].eval()

    if not hparams["test_only"]:
        Interpreter_brain.fit(
            epoch_counter=Interpreter_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    Interpreter_brain.checkpointer.recover_if_possible(
        min_key="loss",
    )

    test_stats = Interpreter_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
