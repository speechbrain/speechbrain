# Sound Classification - ESC50 Dataset

This recipe trains a classifier (`train_classifier.py`) for the ESC50 multiclass sound classification dataset. This classification is mainly adapted from the Speechbrain UrbanSound8k recipe.

The classification recipe makes use of a [CNN14 model](https://arxiv.org/abs/1912.10211) and a convolutional encoder pretrained on the [VGG Sound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset with self-supervised learning. The scripts offer the possibility to train both with log-spectra and log-mel audio features.

We have two main training scripts. Here's the breakdown, and how to run them:


## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

## Training script for CNN14 model
This script trains a CNN14 model on the ESC50 dataset. To run this, you can use the command

`python train_classifier.py hparams/cnn14_classifier.yaml --data_folder /yourpath/ESC50`.

An example training run can be found in [here](https://www.dropbox.com/sh/fbe7l14o3n8f5rw/AACABE1BQGBbX4j6A1dIhBcSa?dl=0).

## Training script for CONV2D model
This script trains a simple convolutional classifier on the ESC50 dataset. To run this, you can use the command

`python train_classifier.py hparams/conv2d_classifier.yaml --data_folder /yourpath/ESC50`.

An example training run can be found in [here](https://www.dropbox.com/sh/tl2pbfkreov3z7e/AADwwhxBLw1sKvlSWzp6DMEia?dl=0).

## Performance and computing times
The CNN14, and conv2d models respectively obtain around 82% accuracy and 75% accuracy on a held-out set.

For CNN14, one epoch on ESC50 takes around 11 seconds. For the conv2d model, one epoch on ESC50 takes 15 seconds.

Both of these numbers are obtained with an NVIDIA RTX 3090 GPU.

## Notes:
  - the recipe automatically downloads the ESC50 dataset. You only need to specify the path to which you would like to download it;
  - all of the necessary models are downloaded automatically for each training script.

## Inference Interface (on HuggingFace)
-The huggingface repository of the CNN14 model with an easy inference interface can be accessed through [our huggingface repository](https://huggingface.co/speechbrain/cnn14-esc50/blob/main/README.md)

# How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:
```shell
python train.py hparams/{hparam_file}.py --data_folder /yourpath/ESC50 --test_only
```

```bibtex
@article{Wang_2022,
	doi = {10.1109/lsp.2022.3229643},
	url = {https://doi.org/10.1109%2Flsp.2022.3229643},
	year = 2022,
	publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
	volume = {29},
	pages = {2607--2611},
	author = {Zhepei Wang and Cem Subakan and Xilin Jiang and Junkai Wu and Efthymios Tzinis and Mirco Ravanelli and Paris Smaragdis},
	title = {Learning Representations for New Sound Classes With Continual Self-Supervised Learning},
	journal = {{IEEE} Signal Processing Letters}
}
```

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
