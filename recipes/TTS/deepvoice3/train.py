"""Recipe for training the DeepVoice3 Text-To-Speech model, a fully-convolutional
attention-based neural text-to-speech (TTS) system

https://arxiv.org/abs/1710.07654

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/TIMIT

Authors
* Artem Ploujnikov 2020
"""
import torch
import torchvision
import sys
import speechbrain as sb
import os
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.checkpoints import torch_save
from torch.utils.data import DataLoader


sys.path.append("..")
from datasets.vctk import VCTK
from speechbrain.lobes.models.synthesis.deepvoice3.dataio import pad_to_length


class DeepVoice3Brain(sb.core.Brain):
    """
    A Brain implementation for the DeepVoice3 text-to-speech model

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_pipeline = DataPipeline(
            static_data_keys=self.hparams.train_pipeline['output_keys'],
            dynamic_items=self.hparams.features_pipeline['steps'],
            output_keys=self.hparams.features_pipeline['output_keys'],
        )
        self.last_outputs = {}
        self.last_batch = None

    def compute_forward(self, batch, stage, incremental=False, single=False):
        """Uses the deepvoice3 model to output the spectrograms of the generated
        speech

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        incremental: bool
            Whether to compute the forward pass incrementally,
            step by step, without a time (used for generation)
        single: bool
            whether to extract only a single element from the batch. This
            is useful for the saving of incremental reconstruction samples,
            which would be impractical to compute on an entire batch during
            training - but more manageable if only a single example is used

        Returns
        -------
        predictions : torch.Tensor
            A tuple of tensors containing the predictions
            Format: (mel_outputs, linear_outputs, alignments, done)

            mel_outputs: the MEL-scale diagram produced by the model
            linear_outputs: the linear problem produced by the model
            alignments: attention layer alignments
            done: the done tensor (the probabilities of decoding being finished
                at a given step)

        """
        batch = batch.to(self.device)
        features = self.compute_features(
            batch, incremental=incremental, single=single)

        pred = self.hparams.model(
            text_sequences=features['text_sequences'],
            mel_targets=(
                None if incremental else features['mel'].transpose(1, 2)),
            text_positions=features['text_positions'],
            frame_positions=(
                None if incremental else features['frame_positions']),
            input_lengths=features['input_lengths'],
            speaker_ids=features.get('speaker_id_enc'),
            speaker_embed=features.get('speaker_embed')
        )

        return pred

    def fit_batch(self, batch):
        """
        Overrides fit_batch to run the NOAM scheduler on each update. See
        the base class for details

        Arguments
        ---------
        batch: PaddedBatch
            the batch

        Returns
        -------
        loss: torch.Tensor
            the loss
        """
        loss = super().fit_batch(batch)
        old_lr, new_lr = self.hparams.lr_annealing(self.optimizer)
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
        self.last_batch = batch
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
        incremental: bool
            whether or not to compute the model pass incrementally (optional, defaults to False)

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        # TODO: Avoid doing this twice
        features = self.compute_features(batch)
        output_mel, output_linear, attention, output_done = predictions
        target_mel = features['mel'].transpose(1, 2)
        target_done = features['done']
        target_linear = features['linear'].transpose(1, 2)
        target_lengths = features['target_lengths']
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
        self._update_last_output(
            output_linear=output_linear,
            target_linear=target_linear,
            output_mel=output_mel,
            target_mel=target_mel
        )

        return loss_stats.loss

    def compute_features(self, batch, incremental=False, single=False):
        """
        Computes features by running a secondary pipeline - on
        the GPU, if available

        Arguments
        ---------
        batch: PaddedBatch
            a padded batch instance
        incremental: bool
            indicates whether an incremental run is being performed.
            In an incremental run, the features pipeline will not be run
            because no spectrograms are necessary - the model will
            construct one "from scratch"
        single: bool
            whether to extract only a single element from the batch. This
            is useful for the saving of incremental reconstruction samples,
            which would be impractical to compute on an entire batch during
            training - but more manageable if only a single example is used

        Returns
        -------
        features: dict
            computed features (see features_pipeline in the YAML
            definition)
        """
        features = batch.as_dict()
        if not incremental:
            features = self.features_pipeline(features)
        features = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in features.items()}
        if single:
            features = {
                key: value[:1] for key, value in features.items()}
        return features

    def on_fit_start(self):
        """
        Executed when training statrs
        """
        super().on_fit_start()
        if self.hparams.progress_samples:
            if not os.path.exists(self.hparams.progress_sample_path):
                os.makedirs(self.hparams.progress_sample_path)
        self.last_loss_stats = {}

    def _update_last_output(self, **kwargs):
        """
        Updates the internal dictionary of output snapshots
        """
        self.last_outputs.update(
            {key: value[0].detach().cpu()
             for key, value in kwargs.items()})

    def _compute_incremental_outputs(self, batch):
        """
        Computes incremental outputs for the purpose of producing snapshots.
        Incremental mode simulates real-life usage where only text inputs
        are used.
        """
        predictions = self.compute_forward(
            batch, sb.Stage.VALID, incremental=True,
            single=False)
        output_mel, output_linear, _, _ = predictions
        self._update_last_output(
            output_mel_incremental=output_mel,
            output_linear_incremental=output_linear)

    def _save_progress_sample(self, epoch):
        """
        Saves a set of spectrogram samples

        Arguments:
        ----------
        epoch: int
            The epoch number
        """
        entries = [
            (f'{key}.png', value)
            for key, value in self.last_outputs.items()]
        for file_name, data in entries:
            self._save_sample_image(file_name, data, epoch)

    def _save_sample_image(self, file_name, data, epoch):
        """
        Saves a single sample image

        Arguments
        ---------
        file_name: str
            the target file name
        data: torch.Tensor
            the image data
        epoch: int
            the epoch number (used in file path calculations)
        """
        target_path = os.path.join(
            self.hparams.progress_sample_path,
            str(epoch))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        effective_file_name = os.path.join(target_path, file_name)
        sample = data.transpose(-1, -2).squeeze()
        torchvision.utils.save_image(sample, effective_file_name)

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
            # Recover the learning rate (it is updated in fit_batch)
            current_lr = self.optimizer.param_groups[-1]['lr']

            # The train_logger writes a summary to stdout and to the logfile.
            train_stats = dict(
                lr=current_lr, **self.last_loss_stats[sb.Stage.TRAIN])
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
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0)
            if output_progress_sample:
                if self.hparams.progress_samples_incremental:
                    self._compute_incremental_outputs(self.last_batch)
                self._save_progress_sample(epoch)

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def save_for_pretrained(self):
        """
        Saves the necessary files for the pretrained model
        """
        pretrainer = self.hparams.pretrainer
        for key, value in pretrainer.loadables.items():
            path = pretrainer.paths[key]
            torch_save(value, path)


def dataset_prep(dataset, hparams, tokens=None):
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
    return SaveableDataLoader(dataset, collate_fn=PaddedBatch,
                              **hparams["dataloader_options"])


class SingleBatchLoader(DataLoader):
    """
    A DataLoader implementation for single batches - used to fit
    a pre-computed batch from another implementation, for testing
    purposes

    Arguments
    ---------
    batch: dict
        a batch
    """
    def __init__(self, batch):
        super()
        self.batch = batch

    def __iter__(self):
        """The iterator method"""
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
        loader: DataLoader
            the inner data loader
        """
        self.loader = loader
        self.num_iterations = num_iterations
        self.batch = None

    def __iter__(self):
        """The iterator method"""
        if self.batch is None:
            self.batch = next(iter(self.loader))
        for _ in range(self.num_iterations):
            yield self.batch


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
    if hparams.get('save_for_pretrained'):
        tts_brain.save_for_pretrained()

    # Implement evaluation (if specified)
    if 'test' in datasets:
        test_stats = tts_brain.evaluate(
            test_set=datasets["test"],
            test_loader_kwargs=hparams["dataloader_options"],
        )
        print("Test statistics:")
        print(test_stats)


if __name__ == '__main__':
    main()
