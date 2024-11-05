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
| 25-10-2024 | train_hf_wavlm.yaml | GreedySearch | XL  | 11.88% | 11.86% | Unavailable\* | Unavailable\* | 8xRTX 3090 |

\*: Unfortunately, we are unable to upload the checkpoints for the WavLM model at this time. We currently don't have plans to remedy this.

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```
