import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from expBrain import expBrain
import pandas as pd
import numpy as np
from preprocess import preprocess_RECOLA
import torch


def main():
    """The full pipeline of training and evaluating a model.

    Example
    -------

    $ python train.py settings.yaml \\
    --emotion_dimension=valence \\
    --feat_size=768 \\
    --w2v2_hub=LeBenchmark/wav2vec2-FR-2.6K-base \\
    --experiment_folder="./Results" \\
    --data_path="./RECOLA_2016"
    """
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Make the experiments deterministic and reproducabile
    np.random.seed(hparams["seed"])
    torch.manual_seed(hparams["seed"])

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = expBrain(
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

    # print(datasets["train"][0])

    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    hparams["batch_size"] = 9
    hparams["dataloader_options"]["batch_size"] = 9
    # # Load the best checkpoint for evaluation
    emo_id_brain.evaluate(
        test_set=datasets["valid"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    hparams["batch_size"] = 9
    hparams["dataloader_options"]["batch_size"] = 9
    # # Load the best checkpoint for evaluation
    try:
        emo_id_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            test_loader_kwargs=hparams["dataloader_options"],
        )
    except BaseException:
        print("Could not perform automatic test set evaluation")

    hparams["batch_size"] = 1
    hparams["dataloader_options"]["batch_size"] = 1
    # # Create output file with predictions
    emo_id_brain.output_predictions_test_set(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Preprocess dataset
    preprocess_RECOLA(
        hparams["data_path_audio"],
        hparams["data_path_arousal"],
        hparams["data_path_valence"],
        hparams["experiment_folder"],
        hparams["data_processed_folder"],
    )

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("arousal_path", "valence_path")
    @sb.utils.data_pipeline.provides(
        "arousal_path", "valence_path", "arousal", "valence"
    )
    def label_pipeline(arousal_path, valence_path):
        yield arousal_path
        yield valence_path
        try:
            arousal = csvReader(arousal_path, ["GoldStandard"])
        except BaseException:
            arousal = np.array([])
        try:
            valence = csvReader(valence_path, ["GoldStandard"])
        except BaseException:
            valence = np.array([])
        yield arousal
        yield valence

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_processed_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=[
                "id",
                "sig",
                "arousal_path",
                "valence_path",
                "arousal",
                "valence",
            ],
        )

    return datasets


def csvReader(filePath, headers, standardize=False):
    """Reads a csv and returns the values for a set of specific headers"""
    df = pd.read_csv(filePath)
    outs = []
    for header in headers:
        out = df[header].to_numpy()
        if standardize:
            out = (out - out.mean(axis=0)) / out.std(axis=0)
        out = np.expand_dims(out, axis=1)
        outs.append(out)
    outs = np.concatenate(outs, 1)
    return outs


if __name__ == "__main__":
    main()
