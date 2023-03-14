# Summary

This recipe implements a classification training script (`train_classifier.py`) for the ESC50 multiclass sound classification dataset. This classification is mainly adapted from the Speechbrain UrbanSound8k recipe. The classification recipe makes use of a [CNN14 model](https://arxiv.org/abs/1912.10211) and a convolutional encoder pretrained on the [VGG Sound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset with self-supervised learning. The scripts offers the possibility to train both with log-spectra and log-mel audio features.

We have two main training scripts. Here's the breakdown, and how to run them:

- *The training script for CNN14 model:* This script trains a CNN14 model on the ESC50 dataset. To run this, you can use the command `python train_classifier.py hparams/cnn14_classifier.yaml --data_folder /yourpath/ESC50`. An example training run can be found in [`https://drive.google.com/drive/folders/1XLleSta8-GE47T7IFMK-i8apHOs-4NNT?usp=share_link`](https://drive.google.com/drive/folders/1XLleSta8-GE47T7IFMK-i8apHOs-4NNT?usp=share_link);

- *The training script for the conv2d-based model:* This script trains an simple convolutional classifier on the ESC50 dataset. To run this, you can use the command `python train_classifier.py hparams/conv2d_classifier.yaml --data_folder /yourpath/ESC50`. An example training run can be found in [`https://drive.google.com/drive/folders/14qanAjkMmsAk4AQeilGCkMJwdJWo1BF7?usp=share_link`](https://drive.google.com/drive/folders/14qanAjkMmsAk4AQeilGCkMJwdJWo1BF7?usp=share_link).

Note that:
  - the recipe automatically downloads the ESC50 dataset. You only need to specify the path to which you would like to download it;
  - all of the necessary models are downloaded automatically for each training script.

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
