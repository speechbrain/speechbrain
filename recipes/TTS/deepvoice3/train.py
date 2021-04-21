import librosa # Temporary
import torch
import torchvision
import sys
import speechbrain as sb
import math
import os
from typing import Collection
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import TextEncoder
from speechbrain.dataio.dataloader import SaveableDataLoader
from torch.utils.data import DataLoader
from speechbrain.dataio.batch import PaddedBatch


sys.path.append("..")
from datasets.vctk import VCTK
from speechbrain.lobes.models.synthesis.deepvoice3.dataio import pad_to_length
from torchaudio import transforms



class DeepVoice3Brain(sb.core.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_forward(self, batch, stage, incremental=False):
        """Predicts the next word given the previous ones.
        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        incremental: Boolean
            Whether to compute the forward pass incrementally, 
            step by step, without a time (used for generation)
        Returns
        -------
        predictions : torch.Tensor
            A tensor containing the posterior probabilities (predictions).
        """
        batch = batch.to(self.device)

        pred = self.hparams.model(
            text_sequences=batch.text_sequences.data, 
            mel_targets=
                None if incremental else batch.mel.data.transpose(1, 2), 
            text_positions=batch.text_positions.data,
            frame_positions=
                None if incremental else batch.frame_positions.data,
            input_lengths=batch.input_lengths.data            
        )

        return pred

    def fit_batch(self, *args, **kwargs):
        loss = super().fit_batch(*args, **kwargs)
        old_lr, new_lr = self.hparams.lr_annealing(self.optimizer)
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
        return loss

    def compute_objectives(self, predictions, batch, stage, incremental=False):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The posterior probabilities from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        incremental
            whether or not to compute the model pass incrementally (optional, defaults to False)

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        output_mel, output_linear, attention, output_done = predictions
        target_mel = batch.mel.data.transpose(1, 2)
        target_done = batch.done.data
        target_linear = batch.linear.data.transpose(1, 2)
        target_lengths = batch.target_lengths.data
        # TODO: Too much transposing going on - optimize
        if incremental:        
            output_mel = pad_to_length(
                output_mel.transpose(1, 2), target_mel.size(1)).transpose(1, 2)
            output_linear = pad_to_length(
                output_linear.transpose(1, 2), target_linear.size(1)).transpose(1, 2)           
            output_done = pad_to_length(
                output_done.transpose(1, 2), target_done.size(1), 1.).transpose(1, 2)

        output_linear = output_linear[:, :target_linear.size(1), :]
        targets = target_mel, target_linear, target_done, target_lengths
        outputs = output_mel, output_linear, attention, output_done, batch.input_lengths
        loss_stats = self.hparams.compute_cost(
            outputs, targets
        )
        
        self.last_loss_stats[stage] = loss_stats.as_scalar()
        (self.last_output_linear, 
         self.last_target_linear, 
         self.last_output_mel, 
         self.last_target_mel) = [
            tensor.detach().cpu()[0]
            for tensor in (
                output_linear, target_linear,
                output_mel, target_mel
            )]
        
        return loss_stats.loss

    
    def on_fit_start(self):
        super().on_fit_start()
        if self.hparams.progress_samples:
            if not os.path.exists(self.hparams.progress_sample_path):
                os.makedirs(self.hparams.progress_sample_path)
        self.last_loss_stats = {}

    def _save_progress_sample(self, epoch):
        entries = [
            ('target_linear.png', self.last_target_linear),
            ('output_linear.png', self.last_output_linear),
            ('target_mel.png', self.last_target_mel),
            ('output_mel.png', self.last_output_mel)
        ]
        for file_name, data in entries:
            self._save_sample_image(file_name, data, epoch)

    def _save_sample_image(self, file_name, data, epoch):
        target_path = os.path.join(
            self.hparams.progress_sample_path,
            str(epoch))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        effective_file_name = os.path.join(target_path, file_name)
        sample = data.squeeze()
        torchvision.utils.save_image(sample, effective_file_name)

    def _pad_output(self, tensor, value=0.):
        padding = self.hparams.decoder_max_positions - tensor.size(2)
        return F.pad(tensor, (0, padding), value=value)

    def log_stats(self, *args, **kwargs):
        """
        Logs statistics to all registered logger.
        All arguments get passed on to the underlying loggers
        """
        for logger in self.hparams.loggers:
            logger.log_stats(*args, **kwargs)

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
        if stage != sb.Stage.TRAIN:
            stats = {
                "loss": [stage_loss],
            }

        # At the end of validation, we can wrote
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(self.optimizer)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            train_stats = dict(
                lr=old_lr, **self.last_loss_stats[sb.Stage.TRAIN])
            train_stats = {key: [value] for key, value in train_stats.items()}
            self.log_stats(
                {"Epoch": epoch},
                train_stats=train_stats,
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            if self.hparams.ckpt_every_epoch or epoch == self.hparams.number_of_epochs:
                meta_stats = {key: value[0] for key, value in stats.items()}
                self.checkpointer.save_and_keep_only(meta=meta_stats, min_keys=["loss"])
            output_progress_sample =(
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0)
            if output_progress_sample:
                self._save_progress_sample(epoch)                

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
        


def dataset_prep(dataset:DynamicItemDataset, hparams, tokens=None):
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
        tokens = hparams['tokens']

    encoder_pipeline = hparams['train_pipeline']
    pipeline = encoder_pipeline['steps']

    for element in pipeline:
        dataset.add_dynamic_item(element)

    dataset.set_output_keys(encoder_pipeline['output_keys'])
    return SaveableDataLoader(dataset, collate_fn=collate_fn, **hparams["dataloader_options"])


class SingleBatchLoader(DataLoader):
    def __init__(self, batch):
        super()
        self.batch = batch

    def __iter__(self):
        yield self.batch
        

class SingleBatchWrapper(DataLoader):
    """
    A wrapper that retrieves one batch from a DataLoader
    and keeps iterating - useful for overfit tests
    """
    def __init__(self, loader: DataLoader, num_iterations=1):
        """
        Class constructor
        
        Arguments
        ---------
        loader
            the inner data loader
        """
        self.loader = loader
        self.num_iterations = num_iterations

    def __iter__(self):
        batch = next(iter(self.loader))
        for _ in range(self.num_iterations):
            yield batch


def collate_fn(examples, *args, **kwargs):
    """
    The collation function, producing padded batches. An exceptional
    behaviour is implemented for 'done' where it will be padded with
    1s instead of 0s. Any additional arguments will be passed on to
    PaddedBatch

    Arguments
    ---------
    examples: list
        the list of examples

    Returns
    -------
    batch: PaddedBatch
        a batch of examples
    """    
    max_done = max(example['done'].shape[0] for example in examples)
    for example in examples:    
        padding_size = max_done - example['done'].shape[0]
        example['done'] = F.pad(example['done'], [0, 0, 0, padding_size], value=1)
    return PaddedBatch(examples, *args, **kwargs)


def dataio_prep(hparams):
    """
    Prepares the datasets using the pipeline

    Arguments
    ---------
    hparams: dict
        pre-parsed HyperPyYAML hyperparameters

    Returns
    -------
    datsets: dict
        
    """
    result = {}
    for name, dataset_params in hparams['datasets'].items():
        # TODO: Add support for multiple datasets by instantiating from hparams - this is temporary
        vctk = VCTK(dataset_params['path']).to_dataset()
        result[name] = dataset_prep(vctk, hparams)

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
    frozen_batch = hparams.get('test_frozen_batch')
    if frozen_batch:
        batch = torch.load(frozen_batch)
        for key in ['mel', 'linear']:
            batch[key] = batch[key].transpose(1, 2)
        loader = SingleBatchLoader(batch)
        datasets = {
            key: loader for key in ['train', 'test']}
    else:
        datasets = dataio_prep(hparams)
        if hparams.get('overfit_test'):
            datasets = {
                key: SingleBatchWrapper(
                    dataset,
                    num_iterations=hparams.get('overfit_test_iterations', 1)) 
                for key, dataset in datasets.items()}
    


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
        valid_set=datasets["valid"],
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