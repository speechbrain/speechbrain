# Speech separation with LibriMix
This folder contains some popular recipes for the [LibriMix Dataset](https://arxiv.org/pdf/2005.11262.pdf) (2/3 sources).

* This recipe supports train with several source separation models on LibriMix, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379), [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975).

Additional dependencies:
```
pip install mir_eval
pip install pyloudnorm
```

To run it:

```
python train.py hparams/sepformer-libri2mix.yaml --data_folder yourpath/Libri2Mix
python train.py hparams/sepformer-libri3mix.yaml --data_folder yourpath/Libri3Mix

```
Note that during training we print the negative SI-SNR (as we treat this value as the loss).


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
