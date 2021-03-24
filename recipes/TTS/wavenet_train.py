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

sys.path.append("..")
from datasets.vctk import VCTK
from common.dataio import audio_pipeline, mel_spectrogram, spectrogram, resample


class WavenetBrain(sb.core.Brain):
    def compute_forward(self, batch, stage, use_targets=True):
        """Predicts the next word given the previous ones.
        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        predictions : torch.Tensor
            A tensor containing the posterior probabilities (predictions).
        """
        batch = batch.to(self.device)
        pred = self.hparams.model(
            mel_targets=batch.mel.data
                if stage == sb.Stage.TRAIN else None,
            frame_positions=batch.frame_positions.data
                if use_targets else None,
            input_lengths=batch.input_lengths.data,
            target_lengths=batch.target_lengths.data
                if use_targets else None
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
        batch = batch.to(self.device)
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


def padded_positions(item_len, max_len):
    """
    Returns a padded tensor of positions
    Arguments
    ---------
    max_len
        the maximum length of a sequence
    item_len
        the total length pof the sequence
    """
    positions = torch.zeros(max_len, dtype=torch.long)
    positions[:item_len] = torch.arange(1, item_len+1, dtype=torch.long)
    return positions


def text_encoder(max_input_len=128, tokens=None):
    """
    Configures and returns a text encoder function for use with the deepvoice3 model
    wrapped in a SpeechBrain pipeline function
    Arguments
    ---------
    max_input_len
        the maximum allowed length of an input sequence
    tokens
        a collection of tokens
    """

    encoder = TextEncoder()
    encoder.update_from_iterable(tokens)
    encoder.add_unk()
    encoder.add_bos_eos()

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("text_sequences", "input_lengths", "text_positions")
    def f(label):
        text_sequence = encoder.encode_sequence_torch(label.upper())
        text_sequence_eos = encoder.append_eos_index(text_sequence)
        input_length = len(label)
        padded_text_sequence_eos = F.pad(
            text_sequence_eos, (0, max_input_len - input_length - 1))
        yield padded_text_sequence_eos.long()
        yield input_length
        yield padded_positions(item_len=input_length, max_len=max_input_len)
        
    return f


def downsample_spectrogram(takes, provides, downsample_step=4):
    """
    A pipeline function that downsamples a spectrogram
    Arguments
    ---------
    downsample_step
        the number of steps by which to downsample the target spectrograms
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(mel):
        mel = mel[:, 0::downsample_step].contiguous()
        return mel
    return f


def pad(takes, provides, length):
    """
    A pipeline function that pads an arbitrary
    tensor to the specified length
    Arguments
    ---------
    takes
        the source pipeline element
    provides
        the pipeline element to output
    length
        the length to which the tensor will be padded
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(x):
        return F.pad(x, (0, length - x.size(-1)))
    return f

#TODO: Remove the librosa dependency
def trim(takes, provides, top_db=15):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)    
    def f(wav):
        x, _ = librosa.effects.trim(wav, top_db=top_db)
        x = torch.tensor(x).to(wav.device)
        return x
    return f

def done(max_output_len=1024, outputs_per_step=1, downsample_step=4):
    @sb.utils.data_pipeline.takes("target_lengths")
    @sb.utils.data_pipeline.provides("done")
    def f(target_length):
        done = torch.ones(max_output_len)
        done[:target_length // outputs_per_step // downsample_step - 1] = 0.
        return done
    
    return f

def frame_positions(max_output_len=1024):
    """
    Returns a pipeline element that outputs frame positions within the spectrogram
    Arguments
    ---------
    max_output_len
        the maximum length of the spectrogram
    """
    range_tensor = torch.arange(1, max_output_len+1)
    @sb.utils.data_pipeline.provides("frame_positions")
    def f():
        return range_tensor
    return f


LOG_10 = math.log(10)

def normalize_spectrogram(takes, provides, min_level_db, ref_level_db):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(linear):
        min_level = torch.tensor(math.exp(min_level_db / ref_level_db * LOG_10)).to(linear.device)
        linear_db = ref_level_db * torch.log10(torch.maximum(min_level, linear)) - ref_level_db
        normalized = torch.clip(
            (linear_db - min_level_db) / -min_level_db,
            min=0.,
            max=1.
        )
        return normalized

    return f


@sb.utils.data_pipeline.takes("mel_downsampled")
@sb.utils.data_pipeline.provides("target_lengths")
def target_lengths(mel):
    return mel.size(-1)


def pad_to_length(tensor: torch.Tensor, length: int, value: int=0.):
    """
    Pads the last dimension of a tensor to the specified length,
    at the end
    
    Arguments
    ---------
    tensor
        the tensor
    length
        the target length along the last dimension
    value
        the value to pad it with
    """
    padding = length - tensor.size(-1)
    return F.pad(tensor, (0, padding), value=value)


OUTPUT_KEYS = [
    "text_sequences", "mel", "input_lengths", "text_positions",
    "frame_positions", "target_lengths", "done", "linear", "linear_raw", "wav"]

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
        # resampling from source sample rate to new sample rate
        resample( 
            orig_freq=hparams['source_sample_rate'],
            new_freq=hparams['sample_rate']),
        # remove leading and trailing silence
        trim(takes="sig_resampled", provides="sig_trimmed"),
        mel_spectrogram(
            takes="sig_trimmed",
            provides="mel_raw",
            n_mels=hparams['mel_dim'],
            n_fft=hparams['n_fft']),
        downsample_spectrogram(
            takes="mel_raw",
            provides="mel_downsampled",
            downsample_step=hparams['mel_downsample_step']),
        normalize_spectrogram(
            takes="mel_downsampled",
            provides="mel_norm",
            min_level_db=hparams['min_level_db'],
            ref_level_db=hparams['ref_level_db']),
        pad(
            takes="mel_norm", provides="mel", length=hparams['max_mel_len']),
        frame_positions(
            max_output_len=hparams['max_mel_len']),
        spectrogram(
            n_fft=hparams['n_fft'],
            hop_length=hparams['hop_length'],
            takes="sig_trimmed",
            provides="linear_raw",
            power=1),
        normalize_spectrogram(
            takes="linear_raw",
            provides="linear_norm",
            min_level_db=hparams['min_level_db'],
            ref_level_db=hparams['ref_level_db']),
        pad(
            takes="linear_norm",
            provides="linear",
            length=hparams['max_output_len']),
        done(max_output_len=hparams['max_mel_len'],
             downsample_step=hparams['mel_downsample_step'],
             outputs_per_step=hparams['outputs_per_step']),
        target_lengths
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
        result[name] = dataset_prep(vctk)
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