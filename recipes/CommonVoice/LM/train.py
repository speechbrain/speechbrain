"""
Recipe  to train  kenlm ngram model  to combine an n-gram with Wav2Vec2.
https://huggingface.co/blog/wav2vec2-with-ngram

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder=/path/to/CommonVoice
Author
 * Pooneh Mousavi 2023
"""

import csv
import os
import sys

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def csv2text():
    """Read CSV file and convert specific data entries into text file."""
    annotation_file = open(
        hparams["train_csv"], "r", newline="", encoding="utf-8"
    )
    reader = csv.reader(annotation_file)
    headers = next(reader, None)
    text_file = open(hparams["text_file"], "w+", encoding="utf-8")
    index_label = headers.index("wrd")
    row_idx = 0
    for row in reader:
        row_idx += 1
        sent = row[index_label]
        text_file.write(sent + "\n")
    text_file.close()
    annotation_file.close()
    logger.info("Text file created at: " + hparams["text_file"])


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from common_voice_prepare import prepare_common_voice  # noqa

    # multi-gpu (ddp) save data preparation
    if not os.path.exists(hparams["text_file"]):
        run_on_main(
            prepare_common_voice,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["output_folder"],
                "train_tsv_file": hparams["train_tsv_file"],
                "accented_letters": hparams["accented_letters"],
                "language": hparams["language"],
                "skip_prep": hparams["skip_prep"],
            },
        )
        csv2text()

    logger.info(f"Start training {hparams['ngram']}-gram kenlm model.")
    tmp_ngram_file = "ngram.arpa"
    cmd = f'lmplz -o {hparams["ngram"]} <"{hparams["text_file"]}" > "{tmp_ngram_file}"'
    os.system(cmd)
    with open(tmp_ngram_file, "r", encoding="utf-8") as read_file, open(
        hparams["ngram_file"], "w", encoding="utf-8"
    ) as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count = line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count) + 1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)
    os.remove(tmp_ngram_file)
    logger.info(
        f"{hparams['ngram']}-gram kenlm model was built and saved in {hparams['ngram_file']}."
    )
