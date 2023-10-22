# Speech separation with WSJ0MIX
This folder contains some popular recipes for the WHAM! and WHAMR! datasets.

* This recipe supports train with several source separation models on WHAM! and WHAMR! datasets, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r ../extra_requirements.txt
```


## How to run:
To run it:

```shell
python train.py hparams/sepformer-wham.yaml --data_folder yourpath/wham_original
python train.py hparams/sepformer-whamr.yaml --data_folder yourpath/whamr
```
Note that during training we print the negative SI-SNR (as we treat this value as the loss).

# How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
python train.py hparams/sepformer-wham.yaml --data_folder yourpath/wham_original --test_only
python train.py hparams/sepformer-whamr.yaml --data_folder yourpath/whamr --test_only
```

# WHAM! and WHAMR! dataset:

* This recipe supports the noisy and reverberant [versions](http://wham.whisper.ai/) of WSJ0 - 2/3 Mix datasets. For WHAM!, simply use `--data_folder /yourpath/wham_original`, and for WHAMR! use `--data_folder /yourpath/whamr`. The script will automatically adjust itself to WHAM and WHAMR, but you must rename the top folder (the folder that contains the `wav8k` subfolder should be named respectively `wham_original` and `whamr`, as the script decides which dataset to use based on the `--data_folder` variable.

* The recipe automatically creates a dataset of room impulse responses (RIRs) from the WHAMR! dataset to use for data augmentation. If you do not this folder for RIR, the `train.py` will automatically create a folder, you just need to specify the path with `--rir_path`. Otherwise you can manually create this dataset using the script in `../meta/create_whamr_rirs.py`.


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training. For this you need to have the WSJ0 dataset (available though LDC at `https://catalog.ldc.upenn.edu/LDC93S6A`). After this the script will automatically convert the sampling frequency and save the files if needed.



# Results

Here are the SI - SNRi results (in dB) on the test set of WHAM!, WHAMR! datasets with SepFormer:


| |SepFormer, WHAM! |
|--- | ---|
|SpeedAugment | 16.3 |
|DynamicMixing | 16.5 |


| | SepFormer. WHAMR! |
| --- | --- |
|NoAugment | 11.4 |
|SpeedAugment | 13.7|
|DynamicMixing | 14.0|

# Training time
It takes about 2h 30 min for WHAMR! (DynamicMixing) and WHAM! on a NVIDIA V100 (32GB).

The output folder with the logs for WHAMR! can be found [here](https://www.dropbox.com/sh/1sia32z01xbfgvu/AADditsqaTyfN3N6tzfEFPica?dl=0).
The output folder with the logs for WHAM! can be found [here](https://www.dropbox.com/sh/sfrgb3xivri432e/AACQodNmiDIKrB9vCeCFUDWUa?dl=0).


# Pretrained Models:
Pretrained models for SepFormer on WHAM!, WHAMR! datasets can be found through huggingface:
* https://huggingface.co/speechbrain/sepformer-wham
* https://huggingface.co/speechbrain/sepformer-whamr

* Pretrained models with the training logs can be found on `https://www.dropbox.com/sh/e4bth1bylk7c6h8/AADFq3cWzBBKxuDv09qjvUMta?dl=0` also.

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/sepformer-whamr).
The 16kHz version of the sepformer can be found [here](https://huggingface.co/speechbrain/sepformer-whamr16k).

# Example calls for running the training scripts

* WHAMR! dataset with dynamic mixing: `python train.py hparams/sepformer-whamr.yaml --data_folder yourpath/whamr --base_folder_dm yourpath/wsj0-processed/si_tr_s --rir_path yourpath/rir_wavs --dynamic_mixing True`

* WHAM! dataset with dynamic mixing: `python train.py hparams/sepformer-wham.yaml --data_folder yourpath/wham_original --base_folder_dm yourpath/wsj0-processed/si_tr_s --dynamic_mixing True`

* WHAMR! dataset without dynamic mixing: `python train.py hparams/sepformer-whamr.yaml --data_folder yourpath/whamr  --rir_path yourpath/rir_wavs`

* WHAM! dataset without dynamic mixing: `python train.py hparams/sepformer-wham.yaml --data_folder yourpath/wham_original`


# Multi-GPU training

You can run the following command to train the model using Distributed Data Parallel (DDP) with 2 GPUs:

```bash
torchrun --nproc_per_node=2 train.py hparams/sepformer-whamr.yaml --data_folder /yourdatapath
```
You can add the other runtime options as appropriate. For more complete information on multi-GPU usage, take a look at this [tutorial](https://colab.research.google.com/drive/13pBUacPiotw1IvyffvGZ-HrtBr9T6l15).


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
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
