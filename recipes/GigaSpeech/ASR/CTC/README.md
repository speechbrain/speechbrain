# Speech Recognition on GigaSpeech with pre-trained self-supervised models and CTC

This folder contains the scripts to finetune any HuggingFace transformer model based
on transformers (wavlm, wav2vec 2, HuBERT...) with CTC for speech recognition on
GigaSpeech. Training can be done on any of the GigaSpeech subset (XL, L, S etc).

## Data access and download

**The XL set is fairly large, 2.2TB are necessary to store the compressed and uncompressed version of the data**

SpeechBrain supports two ways of dealing with the GigaSpeech dataset:
1. [HuggingFace dataset](https://huggingface.co/datasets/speechcolab/gigaspeech/). For HuggingFacem note that **you must use** the HuggingFace client to log in first before running the recipe.
2. [Original Github](https://github.com/SpeechColab/GigaSpeech).

You simply need to follow the instructions on either of the above links. **We strongly
recomment using HuggingFace as the download speed for people outside of China is
much quicker**.

## Data preparation

**This step can be very long depending on your internet connection and filesystem for the XL split of GigaSpeech. For DDP (multi GPU) the recipe must be run once without DDP otherwise it will timeout. You do not want to let X GPUs hang out without doing nothing for hours anyway. Use the *data_prep_only* flag from the yaml to exit after data preparation**

SpeechBrain will automatically download the dataset if you use HuggingFace. Note that if you use HuggingFace, the *data_folder* argument is used to store the **extracted** dataset. However, HuggingFace first needs to download the compressed data, and this is not stored in *data_folder* by default. Indeed, HuggingFace is a bit strict in the way it operates with dataset, and the data will be put into the folder specified by the environment variable *HF_HUB_CACHE* or, if not set, *HF_HOME* or, if not set, *XDG_CACHE_HOME*. Hence, we recommend setting the *HF_HUB_CACHE* to the place where you want to store the data first. For example, you can set it like this:

```export HF_HUB_CACHE=/path/to/your/data/folder```

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# How to run

With a single GPU:
```
python train_with_wavlm.py hparams/file.yaml
```
With multiple GPUs:
```
torchrun --nproc_per_node=8 train_with_wavlm.py hparams/file.yaml
```

# KenLM n-gram CTC rescoring
To enable n-gram rescoring during the decoding, you must download (or train yourself) the n-gram language model:

```
wget https://huggingface.co/wgb14/gigaspeech_lm/resolve/main/3gram_pruned_1e7.arpa.gz
wget https://huggingface.co/wgb14/gigaspeech_lm/resolve/main/4gram.arpa.gz
gunzip -c 3gram_pruned_1e7.arpa.gz > 3gram_pruned_1e7.arpa
gunzip -c 4gram.arpa.gz > 4gram.arpa
```

Then simply modify the *test_beam_search* in the yaml by adding *kenlm_model_path:* and your path as a parameter.

# Rescoring with a Neural Language Model
This can be done by modifying the current recipe. We invite you to have a look at our LibriSpeech CTC recipe for many different examples.

# Results

| Release | Hyperparams file | Decoding method | Finetuning Split | Test WER | Dev WER |  HuggingFace link | Full model link | Training GPUs |
|:-------------:|:---------------------------:|  :----------:|  :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|
| 05-08-23 | train_hf_wavlm.yaml | GreedySearch | XL  | xx | xx | TBD | TBD | 4xRTX 3090 |