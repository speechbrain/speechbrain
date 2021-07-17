import numpy as np
import torch
import sys
import os
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import ast
from process_data import midi_to_pianoroll, piano_roll_to_csv
import pickle
from speechbrain.utils.data_utils import download_file


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

        loss = torch.nn.functional.binary_cross_entropy(predictions, truth)

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
        piano_roll = gen_to_piano_roll(notes)
        binary_roll = roll_to_binary(piano_roll)

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
        piano_roll : list
            List of piano roll tensors
        """
        piano_roll = []
        for i in range(len(notes)):
            if notes[i] == "":
                notes[i] = "0"
            piano_roll_entry = list(map(int, notes[i].split(",")))

            piano_roll_tensor = np.array(piano_roll_entry)
            piano_roll.append(piano_roll_tensor)

        return piano_roll

    def roll_to_binary(piano_roll):
        """This function takes a piano roll and creates a binary vector of size 88
        Arguments
        ---------
        piano_roll : list
            List of string notes
        Returns
        -------
        binary_roll : tensor
            Torch tensor containing a sequence of binarized piano rolls
        """
        binary_roll = torch.zeros(
            (len(piano_roll), 88), dtype=torch.float32
        ).to(run_opts["device"])
        for i in range(len(piano_roll)):
            for j in range(len(piano_roll[i])):
                if piano_roll[i][j] != 0:
                    binary_roll[i][piano_roll[i][j] - 21] = 1
        return binary_roll

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # Set outputs to add into the batch. The batch variable will contain
    # all these fields (e.g, batch.binary_roll_bos, batch.binary_roll_eos)
    sb.dataio.dataset.set_output_keys(
        datasets, ["binary_roll_bos", "binary_roll_eos"],
    )
    return train_data, valid_data, test_data


def return_DL_link(dataset_name):
    if dataset_name == "MAESTRO_v2":
        DL_link = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
    elif dataset_name == "MAESTRO_v3":
        DL_link = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    elif dataset_name == "JSB_chorales":
        DL_link = (
            "http://www-ens.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle"
        )
    elif dataset_name == "Piano-Midi":
        DL_link = (
            "http://www-ens.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle"
        )
    elif dataset_name == "Nottingham":
        DL_link = "http://www-ens.iro.umontreal.ca/~boulanni/Nottingham.pickle"
    elif dataset_name == "MuseData":
        DL_link = "http://www-ens.iro.umontreal.ca/~boulanni/MuseData.pickle"
    else:
        raise ValueError(
            "The dataset name you entered is not supported. Supported datasetnames are: MAESTRO_v2, MAESTRO_v3, JSB_chorales, Piano-Midi, Nottingham, MuseData"
        )

    return DL_link


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

    # check if the csv files exist, and if not create new ones
    train_csv_exists = True if os.path.isfile(hparams["train_csv"]) else False
    valid_csv_exists = True if os.path.isfile(hparams["valid_csv"]) else False
    test_csv_exists = True if os.path.isfile(hparams["test_csv"]) else False

    # set the names to name the downloaded file
    if hparams["dataset_name"] in ["MAESTRO_v2", "MAESTRO_v3"]:
        data_savepath = hparams["data_path"] + ".zip"
    else:
        data_savename = hparams["data_path"].split("/")[-1] + ".pickle"
        data_savepath = os.path.join(hparams["data_path"], data_savename)

    # download the data
    if not os.path.exists(hparams["data_path"]):
        DL_link = return_DL_link(hparams["dataset_name"])
        download_file(DL_link, data_savepath)

        if hparams["dataset_name"] in ["MAESTRO_v2", "MAESTRO_v3"]:
            os.system(
                "unzip {} -d {}".format(data_savepath, hparams["data_path"])
            )

            if hparams["dataset_name"] == "MAESTRO_v2":
                os.system(
                    "mv {} {}".format(
                        os.path.join(
                            hparams["data_path"], "maestro-v2.0.0", "*"
                        ),
                        hparams["data_path"],
                    )
                )
            elif hparams["dataset_name"] == "MAESTRO_v3":
                os.system(
                    "mv {} {}".format(
                        os.path.join(
                            hparams["data_path"], "maestro-v3.0.0", "*"
                        ),
                        hparams["data_path"],
                    )
                )
            else:
                raise ValueError("Unsupported MAESTRO dataset name")

    if not (train_csv_exists and valid_csv_exists and test_csv_exists):
        # if we work with MAESTRO
        if hparams["dataset_name"] in ["MAESTRO_v2", "MAESTRO_v3"]:
            split_songs = [
                ("train", hparams["MAESTRO_params"]["num_train_files"]),
                ("valid", hparams["MAESTRO_params"]["num_valid_files"]),
                ("test", hparams["MAESTRO_params"]["num_test_files"]),
            ]
            datasets = {}
            for split, songs in split_songs:
                datasets[split] = midi_to_pianoroll(split, songs, hparams)
        else:
            # download the dataset in original format if it doesn't exist on data_path
            datasets = pickle.load(open(data_savepath, "rb"))

        for dataset in datasets:
            piano_roll_to_csv(datasets[dataset], dataset, hparams)

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
