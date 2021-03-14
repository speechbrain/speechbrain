import torch
import sys
import speechbrain as sb
from typing import Collection
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import TextEncoder

sys.path.append("..")
from datasets.vctk import VCTK
from common.dataio import audio_pipeline, mel_spectrogram, spectrogram, resample


class DeepVoice3Brain(sb.core.Brain):
    def compute_forward(self, batch, stage):
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
            text_sequences=batch.text_sequences.data, 
            mel_targets=batch.mel.data if stage == sb.Stage.TRAIN else None, 
            text_positions=batch.text_positions.data,
            frame_positions=batch.frame_positions.data
                if stage == sb.Stage.TRAIN else None,
            input_lengths=batch.input_lengths.data,
            target_lengths=batch.target_lengths.data
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
        batch = batch.to(self.device)

        output_mel, output_lienar, _, output_done, output_lengths = predictions
        target_mel = batch.mel.data
        target_done = batch.done.data
        target_linear = batch.linear.data
        target_lengths = batch.target_lengths.data

        outputs = target_mel, target_linear, target_done, target_lengths
        targets = output_mel, output_lienar, output_done, output_lengths
        loss = self.hparams.compute_cost(
            outputs, targets
        )
        return loss


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

        # At the end of validation, we can wrote
        if stage == sb.Stage.VALID:

            # Update learning rate
            # TODO: Bring this back
            old_lr, new_lr = self.hparams.lr_annealing(self.optimizer)
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

#TODO: This is temporary. Add support for different characters for different languages
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!-'

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
        text_sequence = encoder.encode_sequence_torch(label)
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


@sb.utils.data_pipeline.takes("mel_downsampled")
@sb.utils.data_pipeline.provides("target_lengths")
def target_lengths(mel):
    return len(mel)


OUTPUT_KEYS = [
    "text_sequences", "mel", "input_lengths", "text_positions",
    "frame_positions", "target_lengths", "done", "linear"]


def data_prep(dataset: DynamicItemDataset, max_input_len=128,
              max_output_len=1024, tokens=None, mel_dim: int=80,
              n_fft:int=1024, sample_rate=22050, mel_downsample_step=4,
              hop_length=256, outputs_per_step=1):
    """
    Prepares one or more datasets for use with deepvoice.

    In order to be usable with the DeepVoice model, a dataset needs to contain
    the following keys

    'wav': a file path to a .wav file containing the utterance
    'label': The raw text of the label

    Arguments
    ---------
    datasets
        a collection or datasets
    
    Returns
    -------
    the original dataset enhanced
    """

    if not tokens:
        tokens = ALPHABET
    
    pipeline = [
        audio_pipeline,
        resample(new_freq=sample_rate),
        mel_spectrogram(
            takes="sig_resampled",
            provides="mel_raw",
            n_mels=mel_dim,
            n_fft=n_fft),
        downsample_spectrogram(
            takes="mel_raw",
            provides="mel_downsampled",
            downsample_step=mel_downsample_step),
        pad(takes="mel_downsampled", provides="mel", length=max_output_len),
        text_encoder(max_input_len=max_input_len, tokens=tokens),
        frame_positions(
            max_output_len=max_output_len),
        spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            takes="sig",
            provides="linear_raw"),
        downsample_spectrogram(
            takes="linear_raw",
            provides="linear_downsampled",
            downsample_step=mel_downsample_step),
        pad(
            takes="linear_downsampled",
            provides="linear",
            length=max_output_len),
        done(max_output_len=max_output_len,
             downsample_step=mel_downsample_step,
             outputs_per_step=outputs_per_step),
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
        result[name] = data_prep(vctk)
    return result


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
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
    tts_brain = DeepVoice3Brain(
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
    tts_brain.fit(
        epoch_counter=tts_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        # TODO: Implement splitting - this is not ready yet
        valid_set=datasets["train"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

# TODO: Add a test set
    # Load the best checkpoint for evaluation
#    test_stats = tts_brain.evaluate(
#        test_set=datasets["test"],
#        min_key="error",
#        test_loader_kwargs=hparams["dataloader_options"],
#    )


if __name__ == '__main__':
    main()