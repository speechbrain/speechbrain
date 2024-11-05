# How to add your recipe in tests/recipes
The folder `tests/recipes` is introduced for tracking all the recipes and their connected resources (e.g., HuggingFace repo, README files, recipe folders, etc).
Each CSV file in that folder corresponds to one recipe dataset and enlists depending recipe tests.

When you write a new recipe (e.g., recipes/your_dataset/) you need to:
1. ensure the CSV file `tests/recipes/your_dataset.csv` exists (simply copy the header from another CSV)
2. add a new line to the `tests/recipes/your_dataset.csv`.

More specifically, you have to fill the following fields:

- Task (mandatory):
    The task that the recipe is addressing (e.g.. `ASR`).
- Dataset (mandatory):
    Dataset of the recipe (e.g. `LibriSpeech`).
- Script_file (mandatory):
    Training script of the recipe (e.g., `recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py`)
- Hparam_file (mandatory):
    Hyperparameter file of the recipe (e.g., `recipes/LibriSpeech/ASR/CTC/hparams/train_with_wav2vec.yaml`)
- Data_prep_file (optional):
    Data preparation file (e.g., `recipes/LibriSpeech/librispeech_prepare.py`)
- Readme_file (mandatory):
    Readme file describing the recipe (e.g., `recipes/LibriSpeech/ASR/CTC/README.md`)
- Result_url (mandatory):
    URL where the output folder is stored (e.g., `https://www.dropbox.com/sh/qj2ps85g8oiicrj/AAAxlkQw5Pfo0M9EyHMi8iAra?dl=0` ).
    Note that with SpeechBrain we would like to make available the full output folder to the users. The output folder contains the logs, checkpoints, etc that help users debug and reproduce the results.
    Make sure this URL is mentioned in the README file.
- HF_repo (optional):
    Link to the HuggingFace repository containing the pre-trained model (e.g., `https://huggingface.co/speechbrain/asr-wav2vec2-librispeech`). If specified, it must be mentioned in the README file.
- test_debug_flags (optional):
    This optional field reports the flags to run recipe tests (see `tests/.run-recipe-tests.sh`). The goal of the recipe tests is to run an experiment with a tiny dataset and 1-make sure it runs 2-make sure it overfits properly.
    For instance, `--data_folder=tests/samples/ASR/ --train_csv=tests/samples/annotation/ASR_train.csv --valid_csv=tests/samples/annotation/ASR_train.csv --test_csv=[tests/samples/annotation/ASR_train.csv] --number_of_epochs=10 --skip_prep=True` will run an experiment with the given train and hparams files using a tiny dataset (ASR_train.csv)
 - test_debug_checks (optional)
     Checks if the recipe test produces the expected output. For instance,`file_exists=[env.log,hyperparams.yaml,log.txt,train_log.txt,train.py,wer_ASR_train.txt,save/lm.ckpt,save/tokenizer.ckpt] performance_check=[train_log.txt, train loss, <350, epoch: 10]` will first checks if the files in the file_exists list have been created and then checks if the training loss reported in train_log.txt is below a certain threshold.
