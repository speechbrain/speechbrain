#!/usr/bin/python3
"""This recipe to train L-MAC to interpret audio classifiers.

The command to run for this recipe, with WHAM augmentation (as used in the L-MAC paper)
    python train_lmac.py hparams/lmac_cnn14.yaml --data_folder=/yourpath/ESC50 --add_wham_noise True --wham_folder=/yourpath/wham_noise

For more details, please refer to the README file.


Authors
    * Francesco Paissan 2024
    * Cem Subakan 2024
"""
import sys

import torch
import torch.nn.functional as F
from esc50_prepare import dataio_prep, prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from interpreter_brain import InterpreterBrain
from wham_prepare import combine_batches, prepare_wham

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.distributed import run_on_main

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


class LMAC(InterpreterBrain):
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
        X_stft_logpower, X_mel, X_stft, _ = self.preprocess(wavs)
        X_stft_phase = spectral_phase(X_stft)

        hcat, _, predictions, class_pred = self.classifier_forward(X_mel)
        if print_probability:
            predictions = F.softmax(predictions, dim=1)
            class_prob = predictions[0, class_pred].item()
            print(f"classifier_prob: {class_prob}")

        xhat = self.modules.psi(hcat).squeeze(1)

        Tmax = xhat.shape[1]
        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
            X_int = xhat * X_stft_logpower[:, :Tmax, :]

        return (
            X_int.transpose(1, 2),
            xhat.transpose(1, 2),
            X_stft_phase,
            X_stft_logpower,
        )

    def compute_forward(self, batch, stage):
        """Forward computation defined for to generate the saliency maps with L-MAC"""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # augment batch with WHAM!
        if hasattr(self.hparams, "add_wham_noise"):
            if self.hparams.add_wham_noise:
                wavs = combine_batches(wavs, iter(self.hparams.wham_dataset))

        X_stft_logpower, X_mel, X_stft, _ = self.preprocess(wavs)

        # Embeddings + sound classifier
        hcat, _, predictions, class_pred = self.classifier_forward(X_mel)

        xhat = self.modules.psi(hcat).squeeze(1)

        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)

        if stage == sb.Stage.VALID:
            # save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                self.viz_ints(X_stft, X_stft_logpower, batch, wavs)

        if stage == sb.Stage.TEST and self.hparams.save_interpretations:
            # During TEST save always, if required
            self.viz_ints(X_stft, X_stft_logpower, batch, wavs)

        return ((wavs, lens), predictions, xhat, hcat)

    def extra_metrics(self):
        """This function defines the extra metrics required for L-MAC.
        This is limited to the counter() function which is used to count the number of data items which passes the crosscorrelation threshold, during the finetuning stage of L-MAC.
        """

        def counter(c):
            return c

        return {"in_masks": counter}

    def compute_objectives(self, pred, batch, stage):
        """Helper function to compute the objectives"""
        (
            batch_sig,
            predictions,
            xhat,
            _,
        ) = pred

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

        Tmax = xhat.shape[1]

        # map clean to same dimensionality
        X_stft_logpower_clean = X_stft_logpower_clean[:, :Tmax, :]

        mask_in = xhat * X_stft_logpower[:, :Tmax, :]
        mask_out = (1 - xhat) * X_stft_logpower[:, :Tmax, :]

        if self.hparams.use_stft2mel:
            X_in = torch.expm1(mask_in)
            mask_in_mel = self.hparams.compute_fbank(X_in)
            mask_in_mel = torch.log1p(mask_in_mel)

            X_out = torch.expm1(mask_out)
            mask_out_mel = self.hparams.compute_fbank(X_out)
            mask_out_mel = torch.log1p(mask_out_mel)

        if self.hparams.finetuning:
            crosscor = self.crosscor(X_stft_logpower_clean, mask_in)
            crosscor_mask = (crosscor >= self.hparams.crosscor_th).float()

            max_batch = (
                X_stft_logpower_clean.view(X_stft_logpower_clean.shape[0], -1)
                .max(1)
                .values.view(-1, 1, 1)
            )
            binarized_oracle = (
                X_stft_logpower_clean >= self.hparams.bin_th * max_batch
            ).float()

            if self.hparams.guidelosstype == "binary":
                rec_loss = (
                    F.binary_cross_entropy(
                        xhat, binarized_oracle, reduce=False
                    ).mean((-1, -2))
                    * self.hparams.g_w
                    * crosscor_mask
                ).mean()
            else:
                temp = (
                    (
                        (
                            xhat
                            * X_stft_logpower[
                                :, : X_stft_logpower_clean.shape[1], :
                            ]
                        )
                        - X_stft_logpower_clean
                    )
                    .pow(2)
                    .mean((-1, -2))
                )
                rec_loss = (temp * crosscor_mask).mean() * self.hparams.g_w

        else:
            rec_loss = 0
            crosscor_mask = torch.zeros(xhat.shape[0], device=self.device)

        mask_in_preds = self.classifier_forward(mask_in_mel)[2]
        mask_out_preds = self.classifier_forward(mask_out_mel)[2]

        class_pred = predictions.argmax(1)
        l_in = F.nll_loss(mask_in_preds.log_softmax(1), class_pred)
        l_out = -F.nll_loss(mask_out_preds.log_softmax(1), class_pred)
        ao_loss = l_in * self.hparams.l_in_w + self.hparams.l_out_w * l_out

        r_m = (
            xhat.abs().mean((-1, -2, -3))
            * self.hparams.reg_w_l1
            * torch.logical_not(crosscor_mask)
        ).sum()
        r_m += (
            tv_loss(xhat)
            * self.hparams.reg_w_tv
            * torch.logical_not(crosscor_mask)
        ).sum()

        mask_in_preds = mask_in_preds.softmax(1)
        mask_out_preds = mask_out_preds.softmax(1)

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
                mask_out_preds,
            )

        self.in_masks.append(uttid, c=crosscor_mask)
        self.acc_metric.append(
            uttid,
            predict=predictions,
            target=labels,
        )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return ao_loss + r_m + rec_loss


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
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
    print("Class Labels:", class_labels)

    if hparams["finetuning"]:
        if hparams["pretrained_interpreter"] is None:
            raise AssertionError(
                "You should specify pretrained model for finetuning."
            )

    Interpreter_brain = LMAC(
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
