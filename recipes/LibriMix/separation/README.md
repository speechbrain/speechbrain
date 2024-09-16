# Speech separation with LibriMix
This folder contains some popular recipes for the [LibriMix Dataset](https://arxiv.org/pdf/2005.11262.pdf) (2/3 sources).

* This recipe supports train with several source separation models on LibriMix, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r ../extra_requirements.txt
```

## How to run
To run it:

```shell
python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix
python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix

```
Note that during training we print the negative SI-SNR (as we treat this value as the loss).

## How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri3Mix --test_only
python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix --test_only
```

# Libri2/3 Mix
* The Dataset can be created using the scripts at `https://github.com/JorisCos/LibriMix`.


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training.

# Results

Here are the SI - SNRi results (in dB) on the test set of LibriMix dataset with SepFormer:

| | SepFormer. Libri2Mix |
| --- | --- |
|SpeedAugment | 20.1|
|DynamicMixing | 20.4|


| | SepFormer. Libri3Mix |
| --- | --- |
|SpeedAugment | 18.4|
|DynamicMixing | 19.0|


# Example calls for running the training scripts

* Libri2Mix with dynamic mixing `python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True`

* Libri3Mix with dynamic mixing `python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True`

* Libri2Mix with dynamic mixing with WHAM! noise in the mixtures `python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True --use_wham_noise True`

* Libri3Mix with dynamic mixing with WHAM! noise in the mixtures `python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix/ --base_folder_dm yourpath/LibriSpeech_processed --dynamic_mixing True --use_wham_noise True`


The output folder with the trained model and the logs can be found [here](https://www.dropbox.com/sh/kmyz7tts9tyg198/AACsDcRwKvelXxEB-k5q1OaIa?dl=0) for 3-speaker mixtures and [here](https://www.dropbox.com/sh/skkiozml92xtgdo/AAD0eJxgbCTK03kAaILytGtVa?dl=0) for 2-speakers ones.

# Multi-GPU training

You can run the following command to train the model using Distributed Data Parallel (DDP) with 2 GPUs:

```bash
torchrun --nproc_per_node=2 train.py hparams/sepformer-libri2mix.yaml --data_folder /yourdatapath
```
You can add the other runtime options as appropriate. For more complete information on multi-GPU usage, take a look at [our documentation](https://speechbrain.readthedocs.io/en/latest/multigpu.html).


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


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
