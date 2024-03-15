"""Recipe for evaluating a speech synthesis model using one or more of the evaluators provided

Authors
* Artem Ploujnikov, 2024
"""

import csv
import json
import logging
import re
import speechbrain as sb
import string
import sys
import torch
from collections import OrderedDict
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.inference.eval import itemize
from pathlib import Path
from torch import nn
from types import SimpleNamespace
from hyperpyyaml import load_hyperpyyaml
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


class Evaluator:
    """Encapsulates the evaluation loop for a TTS evaluation
    model

    Arguments
    ---------
    hparams : dict
        Raw hyperparameters
    run_opts : dict
        The run options
    """

    def __init__(
        self, hparams, run_opts=None,
    ):
        self.hparams = SimpleNamespace(**hparams)
        self.run_opts = run_opts or {}
        self.device = run_opts.get("device", "cpu")
        modules = hparams.get("modules")
        self.tts = self.hparams.tts(run_opts={"device": self.device})
        self.modules = (
            nn.ModuleDict(self.hparams.modules).to(self.device)
            if modules
            else {}
        )
        self.enabled_evaluators = set(self.hparams.evaluations.split(","))
        self.evaluators = {
            evaluator_key: evaluator_fn(run_opts={"device": self.device})
            for evaluator_key, evaluator_fn in self.hparams.evaluators.items()
            if evaluator_key in self.enabled_evaluators
        }

    def evaluate(self, dataset):
        """Runs the evaluation loop on the specified dataset

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            the dataset
        """

        self.on_evaluate_start()
        dataloader = make_dataloader(
            dataset, batch_size=self.hparams.batch_size
        )
        for batch in tqdm(dataloader, desc="Evaluation"):
            self.evaluate_batch(batch)
        self.on_evaluate_end()

    def on_evaluate_start(self):
        """Invoked at the beginning of evaluation"""
        self.evaluators = {}
        self.output_files = {}
        self.output_writers = {}
        self.details = {}
        for key, evaluator_fn in self.hparams.evaluators.items():
            self.evaluators[key] = evaluator_fn(
                run_opts={"device": self.device}
            )
            self.init_evaluator_result(key)

    def init_evaluator_result(self, evaluator_key):
        """Opens the CSV file to which evaluation results will be written
        and outputs the header

        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        """
        file_name = self.hparams.output_files[evaluator_key]
        output_path = Path(file_name).parent
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_file = open(file_name, "w")
        columns = self.get_report_columns(evaluator_key)
        self.output_writers[evaluator_key] = csv.DictWriter(
            self.output_file, columns
        )
        self.output_writers[evaluator_key].writeheader()
        self.details[evaluator_key] = []

    def on_evaluate_end(self):
        """Invoked at the end of evaluation"""
        self.flush()
        self.close()
        self.write_summary()

    def flush(self):
        """Flushes all output files to disk"""
        for output_file in self.output_files.values():
            output_file.flush()

    def close(self):
        """Closes all output files"""
        for output_file in self.output_files.values():
            output_file.close()

    def evaluate_batch(self, batch):
        """Runs the evaluaion on a single batch

        Arguments
        ---------
        batch : PaddedBatch
            A single item (wrapped in a batch)
        """
        batch = batch.to(self.device)
        wav, length = self.synthesize(batch.label_norm)
        for evaluator_key, evaluator in self.evaluators.items():
            result = evaluator.evaluate(
                wav,
                length,
                text=batch.label_norm_eval,
                wavs_ref=batch.sig.data,
                length_ref=batch.sig.lengths,
            )
            result_items = itemize(result)
            self.write_result(evaluator_key, batch.uttid, result_items)
            self.details[evaluator_key].extend(result_items)
        self.flush()

    def synthesize(self, text):
        """Calls the TTS system to synthesize audio from text

        Arguments
        ---------
        text : str
            The text to be synthesized

        Returns
        -------
        wav : torch.Tensor
            The waveform
        length : torch.Tensor
            The lengths
        """
        tts_out = self.tts(text)
        wav, length = self.modules.tts2wav(tts_out)
        if wav.dim() > 2:
            wav = wav.squeeze(1)
        return wav, length

    def write_result(self, key, item_ids, result):
        """Outputs a speech evaluation result to the target file

        Arguments
        ---------
        key : str
            The evaluator key
        item_id : list
            A list of IDs
        result : list
            speechbrain.inference.eval.SpeechEvaluationResult
            The evaluation result from a single evaluator"""
        writer = self.output_writers[key]
        for item_id, item_result in zip(item_ids, result):
            row = {
                "id": item_id,
                "score": item_result.score,
                **item_result.details,
            }
            writer.writerow(flatten(row))

    def get_report_columns(self, evaluator_key):
        """Returns the columns for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            the identifier of the evaluator

        Returns
        -------
        columns : list[str]
            a list of column headers
        """
        bogus_wavs = torch.randn(2, 10000, device=self.device)
        bogus_length = torch.tensor([1.0, 1.0], device=self.device)
        evaluator = self.evaluators[evaluator_key]
        result = evaluator.evaluate(
            wavs=bogus_wavs,
            length=bogus_length,
            text=["BOGUS"] * len(bogus_wavs),
            wavs_ref=bogus_wavs,
            length_ref=bogus_length,
        )
        return list(
            OrderedDict.fromkeys(["id", "score"] + list(result.details.keys()))
        )

    def compute_summary(self):
        """Computes the summarized statistics"""
        return {
            f"{evaluator_key}_{stat_key}": value
            for evaluator_key in self.enabled_evaluators
            if evaluator_key in self.details
            for metric_key in self.hparams.eval_summary[evaluator_key][
                "descriptive"
            ]
            for stat_key, value in descriptive_statistics(
                items=self.details[evaluator_key], key=metric_key,
            ).items()
        }

    def write_summary(self):
        """Outputs summarized statistics"""
        summary = self.compute_summary()
        file_name = Path(self.hparams.output_files["summary"])
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)


def dataio_prepare(hparams):
    """Prepares the dataset

    Arguments
    ---------
    hparams : dict
        Raw hyperparameters"""

    data_folder = hparams["data_folder"]
    eval_dataset = hparams["eval_dataset"]
    json_path = hparams[f"{eval_dataset}_json"]

    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=json_path,
        replacements={"data_root": data_folder},
        output_keys=["uttid", "label"],
    )
    dataset.add_dynamic_item(label_norm_pipeline)
    dataset.add_dynamic_item(audio_ref_pipeline)
    dataset.set_output_keys(
        ["uttid", "label_norm_eval", "label_norm", "label_norm_length", "sig"]
    )

    if hparams["sorting"] == "ascending":
        dataset = dataset.filtered_sorted(sort_key="label_norm_length")
    elif hparams["sorting"] == "descending":
        dataset = dataset.filtered_sorted(
            sort_key="label_norm_length", reverse=True
        )
    return dataset


def flatten(value):
    """Converts tensors to scalars and lists of strings to strings

    Arguments
    ---------
    value : dict
        the dictionary to flatten

    Returns
    -------
    result : dict
        a flattened dictionary
    """
    return {
        key: item_value.item() if torch.is_tensor(item_value) else item_value
        for key, item_value in value.items()
    }


def descriptive_statistics(items, key):
    """Computes descriptive statistics for the summary

    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        """
    values = torch.tensor([item.details[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{key}_{stat_key}": value.item() for stat_key, value in stats.items()
    }


RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


@sb.utils.data_pipeline.takes("label")
@sb.utils.data_pipeline.provides(
    "label_norm", "label_norm_length", "label_norm_eval"
)
def label_norm_pipeline(label):
    """Normalizes labels for ASR comparison, converting to uppercase and removing
    punctuation

    Arguments
    ---------
    label : str
        The unnormalized label

    Returns
    -------
    result : str
        The normalized label
    """
    label_norm = label.upper()
    yield label_norm
    yield len(label_norm)
    label_norm_eval = RE_PUNCTUATION.sub("", label_norm)
    yield label_norm_eval


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_ref_pipeline(wav):
    """The audio loading pipeline for references

    Arguments
    ---------
    wav : str
        The file path

    Returns
    -------
    sig : torch.Tensor
        The waveform
    """
    sig = sb.dataio.dataio.read_audio(wav)
    return sig


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from ljspeech_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["prepare_save_folder"],
            "splits": hparams["splits"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
            "skip_ignore_folders": hparams["skip_ignore_folders"],
            "frozen_split_path": hparams["frozen_split_path"],
        },
    )

    dataset = dataio_prepare(hparams)

    evaluator = Evaluator(hparams, run_opts)
    evaluator.evaluate(dataset)
