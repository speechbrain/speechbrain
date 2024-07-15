
# K-means (Quantization)
This folder contains recipes for training K-means clustering model for the LJSpeech Dataset.
The model serves to quantize self-supervised representations into discrete representation. Thus representations can be used as a discrete audio input for various tasks including classification, ASR and speech generation.
It supports  kmeans model using the features from  HuBERT, WAVLM or Wav2Vec.

You can download LibriSpeech at http://www.openslr.org/12

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# How to run:
To configure the SSL model type and corresponding Hub in your YAML configuration file, follow these steps:

1. Locate the `model_config` section in your YAML file.
2. Modify the `ssl_model_type` field to specify one of the SSL models: "Hubert", "WavLM", or "Wav2Vec2".
3. Update the `ssl_hub` field with the specific name of the SSL Hub associated with your chosen model type.

```shell
python train.py hparams/train_discrete_ssl.yaml
```

# Results

The checkpoints can be found at [this](https://huggingface.co/speechbrain/SSL_Quantization) HuggingFace repository.



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