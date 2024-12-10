# AudioMNIST Dataset classification
This folder contains recipes for spoken digits recognition with [AudioMNIST Dataset](https://github.com/soerenab/AudioMNIST),
including sample recipes for the [Learnable gammatone filterbank audio frontend](https://septentrio.uit.no/index.php/nldl/article/view/6279).
The recipes include the original [AudioNet architecture](https://arxiv.org/abs/1807.03418) and two other versions using a learnable
Gammatone filter bank as frontend.

# How to run
To run it, please type:

```
python train.py hparams/audionet.yaml --data_folder=/path_to_/AudioMNIST (AudioNet)
# LFB frontends
python train.py hparams/audionet_lfb.yaml --data_folder=/path_to_/AudioMNIST --seed=1234 (AudioNet with Gammatone LFB)
python train.py hparams/audionet_custom_lfb.yaml --data_folder=/path_to_/AudioMNIST --seed=1234 (customized AudioNet with Gammatone LFB)
```

# Performance summary

[Test accuracy on AudioMNIST split 0]
| System | Accuracy |
|---------------------- | ------------ |
| AudioNet | 94.05% |
| AudioNet + LFB | 97.03% |
| AudioNet custom + LFB | 96.70% |


# Checkpoints and Training logs

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
- AudioNet: https://os.unil.cloud.switch.ch/swift/v1/lts2-speechbrain/AudioMNIST/results/audionet
- AudioNet + LFB: https://os.unil.cloud.switch.ch/swift/v1/lts2-speechbrain/AudioMNIST/results/audionet_lfb
- AudioNet custom + LFB: https://os.unil.cloud.switch.ch/swift/v1/lts2-speechbrain/AudioMNIST/results/audionet_custom_lfb

## Notes

- The recipe automatically downloads the AudioMNSIT dataset. You only need to specify the path to which you would like to download it.

- The dataset has 5 different splits for training/validation/test samples. Check the yaml recipe for more information.

---------------------------------------------------------------------------------------------------------

## Citing

If you find this recipe useful, please cite:

```bibtex
@inproceedings{learnablefb,
      title = {Learnable filter-banks for CNN-based audio applications},
      author = {Peic Tukuljac, Helena and Ricaud, Benjamin and Aspert,  Nicolas and Colbois, Laurent},
      journal = {Proceedings of the Northern Lights Deep Learning Workshop  2022 },
      series = {Proceedings of the Northern Lights Deep Learning Workshop.  3},
      pages = {9},
      year = {2022},
      abstract = {We investigate the design of a convolutional layer where  kernels are parameterized functions. This layer aims at  being the input layer of convolutional neural networks for  audio applications or applications involving time-series.  The kernels are defined as one-dimensional functions having  a band-pass filter shape, with a limited number of  trainable parameters. Building on the literature on this  topic, we confirm that networks having such an input layer  can achieve state-of-the-art accuracy on several audio  classification tasks. We explore the effect of different  parameters on the network accuracy and learning ability.  This approach reduces the number of weights to be trained  and enables larger kernel sizes, an advantage for audio  applications. Furthermore, the learned filters bring  additional interpretability and a better understanding of  the audio properties exploited by the network.},
      url = {https://septentrio.uit.no/index.php/nldl/article/view/6279},
      doi = {10.7557/18.6279},
}
```

If you use **SpeechBrain**, please cite:

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

---------------------------------------------------------------------------------------------------------

