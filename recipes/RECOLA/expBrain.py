import os
import speechbrain as sb
import torch
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm
import numpy as np
import pandas as pd
from speechbrain.lobes.features import Fbank


def calc_CCC(tensor1, tensor2):
    """Calculating Concordance Correlation Coefficient (CCC) using pytorch
    This method allows backpropagation and being used as the loss function

    Arguments
    ---------
    tensor1 : torch.FloatTensor
        one dimensional torch tensor representing the list of values for the first tensor.
    tensor2 : torch.FloatTensor
        one dimensional torch tensor representing the list of values for the first tensor.

    Returns
    -------
    torch.FloatTensor
        Single value torch tensor for calculated CCC

    Example
    -------
    >>> ccc = calc_CCC(torch.rand(50), torch.rand(50))
    """
    mean_gt = torch.mean(tensor2, 0)
    mean_pred = torch.mean(tensor1, 0)
    var_gt = torch.var(tensor2, 0)
    var_pred = torch.var(tensor1, 0)
    v_pred = tensor1 - mean_pred
    v_gt = tensor2 - mean_gt
    denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
    cov = torch.mean(v_pred * v_gt)
    numerator = 2 * cov
    ccc = numerator / denominator
    return ccc


def loss_ccc(prediction, ground_truth):
    """Calculating loss between the predictions and groundtrouths using CCC"""
    prediction = prediction.view(-1, prediction.size()[1])
    ground_truth = ground_truth.view(-1, ground_truth.size()[1])
    cccs = []
    ccc = calc_CCC(prediction.view(-1), ground_truth.view(-1))
    cccs = [ccc]
    return 1 - torch.stack(cccs)


class CCC_Loss(torch.nn.Module):
    """The torch nn module based class for using CCC as the loss.
    This class allows for better encapsulation than using the CCC loss function.
    """

    def __init__(self):
        super(CCC_Loss, self).__init__()

    def forward(self, prediction, ground_truth):
        loss = loss_ccc(prediction, ground_truth)
        loss = torch.mean(loss)
        return loss


class expBrain(sb.Brain):
    """The brain class for our experiments of emotion recognition
    on RECOLA dataset using wav2vec2 features.
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline to calculate the output of the forward pass."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.get_features(wavs)
        if self.hparams.feature_type == "MFB":
            # print("lengths?", lens, torch.tensor([feats.size()[1]]).float())
            feats = self.modules.mean_var_norm(
                feats, lens
            )  # , torch.tensor([feats.size()[1]]).float()

        embeddings, _ = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)
        outputs = self.modules.tanh(outputs)

        return outputs

    def get_features(self, wavs):
        """Extracting features method
        This method can either extract MFB features or wav2vec2 representations
        depending on the feature_type indicated in hparams following speechbrain paradigm of experimentation"""
        if self.hparams.feature_type == "MFB":
            MFB = Fbank(n_mels=self.hparams.feat_size)
            feats = MFB(wavs.float()).to(self.device).float()
        if self.hparams.feature_type == "W2V2":
            feats = self.w2v2_Chunker(wavs.float())
        return feats

    def w2v2_Chunker(self, wavs):
        """Chunking wav files input before extracting wav2vec2 representations
        to avoid memory issues for files with long duration such as audio files
        for RECOLA dataset that are 5 minutes long, whereas wav2vec2 models are
        usually trained for files with durations of 10 to 30 seconds.
        """
        n_center = self.hparams.wav2vec2_center_chunk * self.hparams.sample_rate
        n_side = self.hparams.wav2vec2_side_chunk * self.hparams.sample_rate
        max_size = wavs.size()[1]
        feats = []
        for i in range(0, max_size, n_center):
            start = i - n_side
            end = i + n_center + n_side
            if start < 0:
                start = 0
            if end > max_size:
                end = max_size
            inp = wavs[:, start:end]
            feat = self.modules.wav2vec2(inp)
            # print("feat_side orig", feat.size()[1])
            if i == 0:
                feat_side = (feat.size()[1] + 1) * n_side / (n_center + n_side)
                feat = feat[:, : -int(feat_side), :]
            elif i + n_center + n_side > max_size:
                feat_side_s = (
                    (feat.size()[1] + 1) * n_side / (n_center + n_side)
                )
                n_side_e = max_size - (i + n_center)
                feat_side_e = (
                    (feat.size()[1] + 1) * n_side_e / (n_center + n_side_e)
                )
                if feat_side_e == 0:
                    feat = feat[:, int(feat_side_s) :, :]
                else:
                    feat = feat[:, int(feat_side_s) : -int(feat_side_e), :]
            else:
                feat_side = (
                    (feat.size()[1] + 1) * n_side / (n_center + 2 * n_side)
                )
                feat = feat[:, int(feat_side) : -int(feat_side), :]
            # print("feat_size", i, inp.size(), feat.size())
            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        # print("FEATS SIZE:",feats.size())
        return feats

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def reshapeTarget(self, tensor, newLen):
        """Reshaping a target tensor to a new length.
        This method is especially usefull for when there are different lengths
        between the predictions and targets.
        During the training the test tensor is reshaped to match the duration of predictions (since resampling would break the backpropagation),
        and during testing the model, the predictions are reshaped to match the duration of targets (since the targets should be untouched for testing).
        """
        tensor = tensor.cpu().numpy()
        oldLen = np.shape(tensor)[1]
        newOut = np.zeros(
            (tensor.shape[0], newLen, tensor.shape[2]), dtype=float
        )
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[2]):
                newOut[i, :, j] = np.interp(
                    np.linspace(0, oldLen, newLen),
                    np.arange(0, len(tensor[i, :, j]), 1),
                    tensor[i, :, j],
                )
        newOut = torch.tensor(newOut).to(self.device)
        return newOut.float()

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        _, lens = batch.sig

        target = batch.arousal
        if self.hparams.emotion_dimension == "valence":
            target = batch.valence
        emoid, _ = target

        emoid = self.reshapeTarget(emoid, predictions.size()[1])

        loss = self.hparams.compute_cost(predictions, emoid)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            if self.hparams.epoch_counter.should_stop(
                current=epoch, current_metric=stage_loss
            ):
                self.hparams.epoch_counter.current = (
                    self.hparams.epoch_counter.limit
                )  # skipping unpromising epochs
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def output_predictions_test_set(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and create output file for predictions of the test set.

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
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )

        self.on_evaluate_start(max_key=max_key, min_key=min_key)  # done before
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1

                emo_ids = batch.id
                output = self.compute_forward(batch, stage=sb.Stage.TEST)
                predictions = (
                    self.reshapeTarget(output, 7501).squeeze(2).cpu().numpy()
                )

                for emo_id, prediction in zip(emo_ids, predictions):
                    outFolder = os.path.join(
                        self.hparams.output_folder,
                        self.hparams.emotion_dimension,
                    )
                    if not os.path.exists(outFolder):
                        os.makedirs(outFolder)
                    savePath = os.path.join(outFolder, emo_id + ".csv")
                    df = pd.DataFrame(prediction, columns=["Prediction"])
                    df.to_csv(savePath, index=False)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
        self.step = 0
