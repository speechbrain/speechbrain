#!/usr/bin/python3
"""Recipe for training a classifier using the
mobvoihotwords Dataset.

To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/xvect.yaml (xvector system)

"""
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from torch.utils.data import DataLoader
from tqdm import tqdm


class SpeakerBrain(sb.core.Brain):
    """Class for GSC training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + command classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # print("wavs.size():{}".format(wavs.size()))

        wav_len = wavs.shape[1]

        win_len = 24000
        if wav_len < win_len:
            zero_pad = torch.zeros((1, win_len - wav_len)).to(self.device)
            # print("zero_pad.size():{}".format(zero_pad.size()))
            wavs = torch.cat((wavs, torch.zeros((1, win_len - wav_len)).to(self.device)), 1)

        # print("lens.size():{}".format(lens.size()))
        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        frame_num = feats.shape[1]

        if self.hparams.use_log1p:
            # Log1p reduces the emphasis on small differences
            feats = torch.log1p(feats)

        compute_cw  = sb.processing.features.ContextWindow(left_frames=75, right_frames=75)
        # print("feats:{}".format(feats.shape))
        feats_contex = compute_cw(feats)
        # print("feats_contex0:{}".format(feats_contex.shape))
        feats_contex = feats_contex.transpose(0, 1)
        # print("feats_contex0:{}".format(feats_contex.shape))
        feats_contex = torch.reshape(feats_contex, (frame_num, 151, 40))
        # print("feats_contex1:{}".format(feats_contex.shape))
        feats_contex = feats_contex[75:-75, :, :]
        # print("feats_contex2:{}".format(feats_contex.shape))
        frame_num = feats_contex.shape[0]
        # noisy_feats = torch.transpose(feats_contex, 0, 1)
        # print("feats_contex:{}".format(feats_contex.shape))
        # print(noisy_feats.shape)

        feats = self.modules.mean_var_norm(feats_contex, torch.ones([frame_num]).to(self.device))

        # Embeddings + classifier
        outputs = self.modules.embedding_model(feats)
        # outputs = self.modules.classifier(embeddings)
        # print("outputs.size():{}".format(outputs.size()))

        # Ecapa model uses softmax outside of its classifer
        # if "softmax" in self.modules.keys():
        #     outputs = self.modules.softmax(outputs)

        output_label = torch.argmax(outputs[:, 0, :], dim=1).cpu().numpy()
        # output_label = np.sum(output_label)
        # print("output_label:{}".format(output_label.shape))
        # print(len(output_label))

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using command-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        command, _ = batch.command_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and self.hparams.apply_data_augmentation:
            command = torch.cat([command] * self.n_augment, dim=0)

        # # compute the cost function
        # # loss = self.hparams.compute_cost(predictions, command, lens)
        # # loss = sb.nnet.losses.nll_loss(predictions, command, lens)

        # if hasattr(self.hparams.lr_annealing, "on_batch_end"):
        #     self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # if stage != sb.Stage.TRAIN:
        #     self.error_metrics.append(uttid, predictions, command, lens)

        keyword1_count = 0
        keyword2_count = 0

        output_label = torch.argmax(predictions[:, 0, :], dim=1).cpu().numpy()

        for t in range(predictions.shape[0]):
            if output_label[t] == 0:
                keyword1_count += 1
            if output_label[t] == 1:
                keyword2_count += 1

        if command == 0:
            if keyword1_count > 0:
                self.result['hixiaowen']['TP'] += 1
            else:
                self.result['hixiaowen']['FN'] += 1
            if keyword2_count > 0:
                self.result['nihaowenwen']['FP'] += 1
            else:
                self.result['nihaowenwen']['TN'] += 1
        if command == 1:
            if keyword1_count > 0:
                self.result['hixiaowen']['FP'] += 1
            else:
                self.result['hixiaowen']['TN'] += 1
            if keyword2_count > 0:
                self.result['nihaowenwen']['TP'] += 1
            else:
                self.result['nihaowenwen']['FN'] += 1

        return None

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
        self.result  = {}
        self.wake_words = ['hixiaowen', 'nihaowenwen']

        for wake_word in self.wake_words:
            self.result[wake_word] = {}
            self.result[wake_word].update({'TP': 0})
            self.result[wake_word].update({'FN': 0})
            self.result[wake_word].update({'FP': 0})
            self.result[wake_word].update({'TN': 0})

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

            if self.hparams.use_tensorboard:
                valid_stats = {
                    "loss": stage_stats['loss'],
                    "ErrorRate": stage_stats["ErrorRate"],
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, valid_stats
                )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            for wake_word in self.wake_words:
                print('result on {}'.format(wake_word))
                compute_metrics(self.result[wake_word])

            # self.hparams.train_logger.log_stats(
            #     {"Epoch loaded": self.hparams.epoch_counter.current},
            #     test_stats=stage_stats,
            # )

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        # print("Epoch loaded: {}".format(self.hparams['epoch_counter'].current))
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                # loss = self.evaluate_batch(batch, stage=Stage.TEST)
                out = self.compute_forward(batch, stage=sb.Stage)
                loss = self.compute_objectives(out, batch, stage=sb.Stage)

                # avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[sb.Stage.TEST, None]
            )
        self.step = 0


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("command")
    @sb.utils.data_pipeline.provides("command", "command_encoded")
    def label_pipeline(command):
        yield command
        command_encoded = label_encoder.encode_sequence_torch([command])
        yield command_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="command",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "command_encoded"]
    )

    return train_data, valid_data, test_data, label_encoder


def compute_metrics(result : dict, verbose=True):

    TP = result['TP']
    FN = result['FN']
    TN = result['TN']
    FP = result['FP']

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    false_positive_rate = FP / (FP + TN) if FP + TN > 0 else 0.0
    false_negative_rate = FN / (FN + TP) if FN + TP > 0 else 0.0

    if verbose:
        print("True Positive:{}".format(result['TP']))
        print("False Negative:{}".format(result['FN']))
        print("True Negative:{}".format(result['TN']))
        print("False Positive:{}".format(result['FP']))
        print("precise:{}".format(precision))
        print("recall:{}".format(recall))
        print("false_positive_rate:{}".format(false_positive_rate))
        print("false_negative_rate:{}".format(false_negative_rate))

    return precision, recall, false_positive_rate, false_negative_rate



if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing GSC and annotation into csv files)
    from prepare_kws import prepare_kws

    # Data preparation
    run_on_main(
        prepare_kws,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # # Training
    # speaker_brain.fit(
    #     speaker_brain.hparams.epoch_counter,
    #     train_data,
    #     valid_data,
    #     train_loader_kwargs=hparams["dataloader_options"],
    #     valid_loader_kwargs=hparams["dataloader_options"],
    # )

    # Load the best checkpoint for evaluation
    test_stats = speaker_brain.evaluate(
        test_set=test_data,
        min_key="ErrorRate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
