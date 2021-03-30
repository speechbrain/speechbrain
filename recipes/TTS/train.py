import librosa
import torch
import sys
import speechbrain as sb
import math
from typing import Collection
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import TextEncoder
import numpy as np

sys.path.append("..")
from datasets.vctk import VCTK
from common.dataio import audio_pipeline, mel_spectrogram, spectrogram, resample

from scipy.signal import firwin, lfilter


class WavenetBrain(sb.core.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_forward(self, batch, stage, use_targets=True):
        # NEED TO FIGURE OUT THE COMPUTE FORWARD STEP
        batch = BatchWrapper(batch).to(self.device)
        
        pred = self.hparams.model(
            mel_targets=batch.mel.data
                if stage == sb.Stage.TRAIN else None
        )
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The posterior probabilities from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = BatchWrapper(batch).to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()
    '''
    Can store class variables if needed at the start of a stage (start of training or start of validating for each epoch)
    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
    '''
    def on_stage_end(self, stage, stage_loss, epoch):
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
            }

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

class BatchWrapper:
    def __init__(self, batch):
        self.batch = batch

    def to(self, device):
        for key, value in self.batch.items():
            if hasattr(value, 'to'):
                self.batch[key] = value.to(device)
        return self
    
    def __getattr__(self, name):
        return self.batch[name]

#TODO: Remove the librosa dependency
def trim(takes, provides, top_db=15):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)    
    def f(wav):
        x, _ = librosa.effects.trim(wav, top_db=top_db)
        x = torch.tensor(x).to(wav.device)
        return x
    return f

def low_cut_filter(takes,provides,cutoff):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)

    def f(wav, fs, cutoff):
        """APPLY LOW CUT FILTER.
        https://github.com/kan-bayashi/PytorchWaveNetVocoder

        Args:
            x (ndarray): Waveform sequence.
            fs (int): Sampling frequency.
            cutoff (float): Cutoff frequency of low cut filter.
        Return:
            ndarray: Low cut filtered waveform sequence.
        """
        if cutoff > 0.0:
            nyquist = fs // 2
            norm_cutoff = cutoff / nyquist
            
            # low cut filter
            fil = firwin(255, norm_cutoff, pass_zero=False)
            lcf_x = lfilter(fil, 1, x)

            return lcf_x
        else:
            return wav

    return f

def mulaw(x, mu=256):
    """Mu-Law companding
    Method described in paper [1]_.
    .. math::
        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])

    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    product = mu*x.abs()
    return x.sign() * product.log1p() / np.log1p(mu)

def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).
    """
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return ((y + 1) / 2 * mu).long()

def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def mulaw_trim(takes,provides,silence_threshold,is_quantized):
    # trim silence in mu-law quantized domain
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)

    def f(wav):
        if is_quantized:
            if silence_threshold > 0:
                # [0, quantize_channels)
                out = mulaw_quantize(wav, 255)
                start, end = start_and_end_indices(out, silence_threshold)
                wav = wav[start:end]

        return wav

    return f

def wav_clip(takes,provides):
    # Clip
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)

    def f(wav):
        return torch.clip(wav, -1.0, 1.0)

    return f

def mulaw_target(takes,provides,is_quantized):
    # return target signal under a mulaw transformation
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)

    def f(wav):
        if is_quantized:
            out = mulaw_quantize(wav, 255)
        else:
            out = mulaw(wav, 255)
        return out

    return f

def zero_pad(takes, provides,n_fft, is_quantized):
    # zero pad
    # this is needed to adjust time resolution between audio and mel-spectrogram
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)

    def f(wav):
        if is_quantized:
            constant_values = mulaw_quantize(torch.tensor(0.0), 255)
        else:
            constant_values = mulaw(torch.tensor(0.0), 255)
        
        l, r = (0,n_fft)
        if l > 0 or r > 0:
            out = F.pad(wav, (l, r), mode="constant", value=constant_values)
        return out

    return f

OUTPUT_KEYS = ["mel", "sig_padded"]

def dataset_prep(dataset:DynamicItemDataset, hparams, tokens=None):
    """
    Prepares one or more datasets for use with wavenet.
    In order to be usable with the Wavenet model, a dataset needs to contain
    the following keys
    'wav': a file path to a .wav file containing the utterance

    Arguments
    ---------
    datasets
        a collection or datasets
    
    Returns
    -------
    the original dataset enhanced
    """

    pipeline = [
        audio_pipeline,
        # remove leading and trailing silence
        trim(
            takes="sig_resampled", 
            provides="sig_trimmed"),
        low_cut_filter(
            takes = "sig_trimmed", 
            provides="sig_cut", 
            cutoff=hparams["highpass_cutoff"]),
        mulaw_trim(
            takes = "sig_cut",
            provides = "sig_silence_trim",
            silence_threshold = hparams["silence_threshold"],
            is_quantized = hparams["is_mulaw_quantized"]),
        mel_spectrogram(
            takes="sig_silence_trim",
            provides="mel",
            hop_length=hparams['hop_length'],
            n_mels=hparams['mel_dim'],
            n_fft=hparams['n_fft'],
            power=1,
            sample_rate=hparams['sample_rate']),
        wav_clip(
            takes = "sig_silence_trim",
            provides = "sig_clipped"),
        mulaw_target(
            takes= "sig_clipped", 
            provides="sig_mulaw", 
            is_quantized = hparams["is_mulaw_quantized"]),
        zero_pad(
            takes = "sig_mulaw",
            provides = "sig_padded",
            n_fft = hparams["n_fft"],
            is_quantized = hparams["is_mulaw_quantized"])
    ]

    for element in pipeline:
        dataset.add_dynamic_item(element)

    dataset.set_output_keys(OUTPUT_KEYS)
    return dataset

def dataio_prep(hparams):
    result = {}
    for name, dataset_params in hparams['datasets'].items():
        # TODO: Add support for multiple datasets by instantiating from hparams - this is temporary
        vctk = VCTK(dataset_params['path']).to_dataset()
        result[name] = dataset_prep(vctk,hparams)
    return result


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)
    
    # Initialize the Brain object to prepare for mask training.
    wavenet_brain = WavenetBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
  
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    wavenet_brain.fit(
        epoch_counter=wavenet_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        # TODO: Implement splitting - this is not ready yet
        valid_set=datasets["train"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )