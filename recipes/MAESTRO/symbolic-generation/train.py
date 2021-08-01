import torch
import sys
import speechbrain as sb
import ast
from hyperpyyaml import load_hyperpyyaml
from prepare_dataset import prepare_dataset


# Brain class for language model training
class MusicLM(sb.core.Brain):
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
        binary_roll, _ = batch.binary_roll_bos

        # Reshape binary roll into 3D tensor
        binary_roll = binary_roll.view(
            binary_roll.shape[0],
            int(binary_roll.shape[1] / self.hparams.emb_dim),
            self.hparams.emb_dim,
        )

        pred, _ = self.hparams.model(binary_roll)

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
        binary_roll, _ = batch.binary_roll_eos

        # Reshape ground truth into 3D tensor
        truth = binary_roll.view(
            binary_roll.shape[0],
            int(binary_roll.shape[1] / self.hparams.emb_dim),
            self.hparams.emb_dim,
        )

        # Get frame level accuracy
        if stage != sb.Stage.TRAIN:
            thresh_values = torch.where(predictions > 0.5, 1, 0)
            flat_truth = torch.flatten(truth)
            flat_pred = torch.flatten(thresh_values)
            TP = torch.sum(
                (flat_truth == flat_pred) * (flat_truth == 1) * (flat_pred == 1)
            )

            negatives_ground_truth = (
                2 * len(flat_truth) / self.hparams.emb_dim
            ) - torch.sum(flat_truth == 1)
            negatives_pred = (
                2 * len(flat_pred) / self.hparams.emb_dim
            ) - torch.sum(flat_pred == 1)
            FN = negatives_pred - negatives_ground_truth

            FP = torch.sum(
                (flat_truth != flat_pred) * (flat_truth == 0) * (flat_pred == 1)
            )
            self.accuracy = TP / (TP + FN + FP)

        loss = self.hparams.compute_cost(predictions, truth)

        return loss

    def fit_batch(self, batch):
        """Runs all the steps needed to train the model on a single batch.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.

        Returns
        -------
        Loss : torch.Tensor
            A tensor containing the loss (single real number).
        """
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # Loss backpropagation (gradient computation)
        (loss / self.hparams.accu_steps).backward()

        # Manage gradient accumulation
        if self.step % self.hparams.accu_steps == 0:

            # Gradient clipping & early stop
            self.check_gradients(loss)

            # Update the parameters
            self.optimizer.step()

            # Reset the gradient
            self.optimizer.zero_grad()

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the start of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if stage != sb.Stage.TRAIN:
            self.accuracy = 0

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
                "log-likelihood": -(stage_loss * self.hparams.emb_dim),
                "accuracy": self.accuracy,
            }

        # At the end of validation, we can wrote
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


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    The language model is trained with the text files specified by the user in
    the hyperparameter file.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : list
        List containing "train", "valid", and "test" sets that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Convert piano roll dataset to DynamicItemDataset
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        hparams["train_csv"]
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        hparams["valid_csv"]
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        hparams["test_csv"]
    )

    datasets = [train_data, valid_data, test_data]

    # Define piano roll processing pipeline. We start from the raw notes and then
    # encode it using a binary piano roll. The binary rolls with bos are used for feeding
    # the neural network, the binary rolls with eos for computing the cost function.
    @sb.utils.data_pipeline.takes("notes")
    @sb.utils.data_pipeline.provides("binary_roll_bos", "binary_roll_eos")
    def text_pipeline(notes):
        # Parse csv line into array

        notes = ast.literal_eval(notes)

        # Create binarized piano roll from input
        binary_roll = gen_to_piano_roll(notes)

        # Flatten training roll for sb pipeline
        binary_roll_bos = binary_roll[: (len(binary_roll) - 1)]
        binary_roll_bos = binary_roll_bos.view(-1)
        yield binary_roll_bos

        # Flatten loss roll for sb pipeline
        binary_roll_eos = binary_roll[1:]
        binary_roll_eos = binary_roll_eos.view(-1)
        yield binary_roll_eos

    def gen_to_piano_roll(notes):
        """This function takes an array of string notes and creates a piano roll

        Arguments
        ---------
        notes : list
            List of string notes

        Returns
        -------
        binary_roll : torch.tensor
            Torch tensor for the binary roll
        """
        binary_roll = torch.zeros(len(notes), 88)

        for i in range(len(notes)):
            if notes[i] == "":
                notes[i] = "0"
            row = list(map(int, notes[i].split(",")))
            binary_roll[i, torch.tensor(row) - 21] = 1

        return binary_roll

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # Set outputs to add into the batch. The batch variable will contain
    # all these fields (e.g, batch.binary_roll_bos, batch.binary_roll_eos)
    sb.dataio.dataset.set_output_keys(
        datasets, ["binary_roll_bos", "binary_roll_eos"],
    )
    return train_data, valid_data, test_data


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create the csv files from the dataset
    prepare_dataset(
        hparams["data_folder"],
        hparams["dataset_name"],
        hparams["train_csv"],
        hparams["valid_csv"],
        hparams["test_csv"],
        hparams,
    )

    # Create dataset objects "train", "valid", and "test"
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Initialize the Brain object to prepare for LM training.
    lm_brain = MusicLM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load best checkpoint for evaluation
    test_stats = lm_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
