# SpeechBrain self-supervised-learning (BEST-RQ streaming and offline) with Libri-Light

This folder contains the script for preparing the Libri-Light dataset, and the script of training a small BEST-RQ model (94M parameters) using Libri-Light. The data preparation requires a few steps that may take a bit of time due to the size and structure of the dataset.

## How to run

1- Download the Libri-Light data (small, or medium, or large split) from

    https://github.com/facebookresearch/libri-light/tree/main/data_preparation


2- Git clone the Libri-Light repo from

    https://github.com/facebookresearch/libri-light

Then, use the ```cut_by_vad.py``` script from the Libri-Light repo to do the VAD of each downloaded split.
For example, if you want to the use the small split, and you want to have most clips after VAD to be 20 seconds, then

    python cut_by_vad.py \
        --input_dir path_to_Libri-Light/small \
        --output_dir Libri-Light_VAD/small_vad \
        --target_len_sec 20

If you also want to use the medium or the large split

    python cut_by_vad.py \
        --input_dir path_to_Libri-Light/medium \
        --output_dir Libri-Light_VAD/medium_vad \
        --target_len_sec 20

    python cut_by_vad.py \
        --input_dir path_to_Libri-Light/large \
        --output_dir Libri-Light_VAD/large_vad \
        --target_len_sec 20

**Note**
   If you want to use more than one split, it is important to save the VAD results of each split into the same folder.
   If you want to use the large split, step 1 and 2 may take days.

3- Libri-Light does not have a dev split. Thus, please use the dev set of another dataset to monitor the training. E.g.,
LibriSpeech dev-clean. In practice, you can just put any wav files in a folder and use the path to this folder as a dev.
Ideally, for this recipe, LibriSpeech dev-clean is in-domain and is therefore a good validation test.


4- Now, you can do the Libri-Light data preparation and train a BEST-RQ model using

    python train.py hparams/BEST-RQ.yaml \
        --data_folder Libri-Light_VAD/ \
        --dev_folder /path/to/LibriSpeech/dev-clean \
        --vad_splits=["small_vad"]  \

or, since this can be a fairly big dataset, one may want to use multiple GPUs

    torchrun --nproc_per_node=8 train.py hparams/BEST-RQ.yaml \
        --data_folder Libri-Light_VAD/ \
        --dev_folder /path/to/LibriSpeech/dev-clean \
        --vad_splits=["small_vad"]  \

To use different amount of training data

```--vad_splits=["small_vad"]``` -> around 600 hours

```--vad_splits=["small_vad", "medium_vad"]``` -> around 6k hours

```--vad_splits=["small_vad", "medium_vad", "large_vad"]``` -> around 60k hours

# Finetuning after pretraining

Please refer to the LibriSpeech [ASR / CTC](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/CTC) recipe for speech recognition finetuning. The produced checkpoint is a standard PyTorch checkpoint and this recipe gives you an example of how to load it. Then, you can
just plug it in any recipe that you are interested in by changing the YAML and train.py a bit!