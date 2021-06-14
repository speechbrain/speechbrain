# Speech separation with WSJ0MIX
This folder contains some popular recipes for the WSJ0MIX task (2/3 sources).

* This recipe supports train with several source separation models on WSJ0-2Mix, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

Additional dependency:
```
pip install mir_eval
```

To run it:

```
python train.py hyperparams/sepformer.yaml --data_folder yourpath/wsj0-mix/2speakers
```
Note that during training we print the negative SI-SNR (as we treat this value as the loss).


# WSJ0-2mix and WSJ0-3mix dataset creation
* The best way to create the datasets is using the original matlab script. This script and the associated meta data can be obtained through the following [link](https://www.dropbox.com/s/gg524noqvfm1t7e/create_mixtures_wsj023mix.zip?dl=1).
* The dataset creation script assumes that the original WSJ0 files in the sphere format are already converted to .wav .


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training. For this you need to have the WSJ0 dataset (available though LDC at `https://catalog.ldc.upenn.edu/LDC93S6A`).


# Results

Here are the SI - SNRi results (in dB) on the test set of WSJ0 - 2/3 Mix with SepFormer:

| | SepFormer, WSJ0-2Mix |
|--- | --- |
|NoAugment | 20.4 |
|DynamicMixing | 22.4 |

| | SepFormer, WSJ0-3Mix |
|--- | --- |
|NoAugment | 17.6 |
|DynamicMixing | 19.8 |


# Training Time
Each epoch takes about 2 hours for WJS0-2MIX and WJS0-2MIX (DynamicMixing ) on a NVIDIA V100 (32GB).

# Pretrained Models:
Pretrained models for SepFormer on WSJ0-2Mix, WSJ0-3Mix, and WHAM! datasets can be found through huggingface:
* https://huggingface.co/speechbrain/sepformer-wsj02mix
* https://huggingface.co/speechbrain/sepformer-wsj03mix

* Pretrained models with the training logs can be found on `https://drive.google.com/drive/u/0/folders/1ZVuROxR711Xib2MsJbcPla4PWqbK1Ddw` also.


You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/sepformer-wsj02mix).

# Example calls for running the training scripts


* WSJ0-2Mix training without dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/2speakers`

* WSJ0-2Mix training with dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/2speakers --base_folder_dm yourpath/wsj0/si_tr_s --dynamic_mixing True`

* WSJ0-3Mix training without dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/3speakers`--num_spks 3

* WSJ0-3Mix training with dynamic mixing `python train.py hparams/sepformer.yaml --data_folder yourpath/wsj0-mix/3speakers`--num_spks 3 --base_folder_dm yourpath/wsj0/si_tr_s --dynamic_mixing True`

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
