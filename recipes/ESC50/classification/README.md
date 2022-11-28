# Summary

This recipe implements the following:

- A classification training script (`train.py`) for the ESC50 multiclass sound classification dataset. This classification is mainly adapted from the Speechbrain UrbanSound8k recipe. The classification recipe makes use of a [CNN14 model](https://arxiv.org/abs/1912.10211) pretrained on the [VGG Sound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset with self-supervised learning.

- [Listen to Interpret](https://arxiv.org/abs/2202.11479v2) which makes use of Non-Negative Matrix Factorization to reconstruct the classifier hidden representation in order to provide an interpretation audio signal for the classifier decision. The corresponding training script is `train_l2i.py`.

- A training script to train an NMF dictionary matrix required for L2I. This script is called `train_nmf.py`.


# PreTrained Model + Easy-Inference


# ESC50 Download and Use


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
