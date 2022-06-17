# The Blind SI-SNR estimation recipe

* The goal of this recipe is to train a neural network to be able to estimate the scale-invariant signal-to-noise ratio (SI-SNR) from the separated signals.

* This model is developed to estimate source separation performance on the REAL-M dataset which consists of real life mixtures.

* The REAL-M dataset can downloaded from [this link](https://sourceseparationresearch.com/static/REAL-M-v0.1.0.tar.gz).

* The paper for the REAL-M dataset can be found on [this arxiv link](https://arxiv.org/pdf/2110.10812.pdf).

* The model is trained with the LibriMix and WHAMR! datasets. You can download LibriMix by following the instructions [here](https://github.com/JorisCos/LibriMix). Instructions on WHAMR! can be found [here](https://wham.whisper.ai/)

# How to Run

* To train with dynamic mixing:

```python train.py hparams/pool_sisnrestimator.yaml --data_folder /yourLibri2Mixpath --base_folder_dm /yourLibriSpeechpath --rir_path /yourpathforwhamrRIRs --dynamic_mixing True --use_whamr_train True --whamr_data_folder /yourpath/whamr --base_folder_dm_whamr /yourpath/wsj0-processed/si_tr_s```

# Results

| Release | hyperparams file | L1-Error (DB) | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 18-10-21 | pool_sisnrestimator.yaml | 1.71 | [HuggingFace](https://huggingface.co/speechbrain/REAL-M-sisnr-estimator) | Not Available| RTX8000 48GB |

You can find the output folders with the training logs [here](https://drive.google.com/drive/folders/1NGncbjvLeGfbUqmVi6ej-NH9YQn5vBmI?usp=sharing).
This [repository](https://huggingface.co/speechbrain/REAL-M-sisnr-estimator-training) provides the Separator models needed to train a blind SI-SNR estimator.

# Training Time
It takes about 5 hours for each epoch on a RTX8000 (48GB).

# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/REAL-M-sisnr-estimator

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing REAL-M**
Please, cite our paper for the REAL-M dataset, if you use it for your research or business.

```bibtex
@misc{subakan2021realm,
      title={REAL-M: Towards Speech Separation on Real Mixtures},
      author={Cem Subakan and Mirco Ravanelli and Samuele Cornell and François Grondin},
      year={2021},
      eprint={2110.10812},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

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
