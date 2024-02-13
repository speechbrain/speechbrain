"""
Recipe  to train subwording tokenization on semantic tokens(Discrete SSL tokens).

To run this recipe, do the following:
> python train.py hparams/train_with_[SSL-model].yaml --data_folder=/path/to/LibriSPeech
Author
 * Pooneh Mousavi 2023
"""

import os
import sys
import logging
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import torchaudio
import csv


logger = logging.getLogger(__name__)

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

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    if not os.path.exists(hparams["tokenized_train"]):
        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "tr_splits": hparams["train_splits"],
                "dev_splits": hparams["dev_splits"],
                "te_splits": hparams["test_splits"],
                "save_folder": hparams["output_folder"],
                "merge_lst": hparams["train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams["skip_prep"],
            },
        )

        with open(hparams["train_csv"], newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            with open(hparams["tokenized_train"], "w", newline="") as csvwrite:
                # writer = csv.writer(csvwrite)
                header = ["id"]
                for layer in hparams["ssl_layer_num"]:
                    header.append(f"textified_tokens_layer_{layer}")
                writer = csv.DictWriter(csvwrite, fieldnames=header)
                writer.writeheader()
                # writer.writerow(header)

                for row in reader:
                    sig = sb.dataio.dataio.read_audio(row[2])
                    info = torchaudio.info(row[2])
                    resampled = torchaudio.transforms.Resample(
                        info.sample_rate, hparams["sample_rate"],
                    )(sig)
                    discrete_unit, _ = hparams["discrete_ssl_model"](
                        resampled.unsqueeze(0),
                        None,
                        ssl_layer_num=hparams["ssl_layer_num"],
                        deduplicte=hparams["deduplicate"],
                    )
                    row_dic = {}
                    row_dic["id"] = row[0]
                    for i, layer in enumerate(hparams["ssl_layer_num"]):
                        tokens = (
                            discrete_unit[:, :, i]
                            - layer * hparams["num_clusters"]
                        ).squeeze(0)
                        tokens_char = " ".join(
                            [chr(token + 97) for token in tokens]
                        )
                        row_dic[f"textified_tokens_layer_{layer}"] = tokens_char
                    writer.writerow(row_dic)

    for layer in hparams["ssl_layer_num"]:
        model_dir = os.path.join(
            hparams["save_folder"], f"tokenizer_layer_{layer}"
        )
        SentencePiece(
            model_dir=model_dir,
            vocab_size=hparams["vocab_size"],
            annotation_train=hparams["tokenized_train"],
            annotation_read=f"textified_tokens_layer_{layer}",
            annotation_format="csv",
            model_type="bpe",
        )
