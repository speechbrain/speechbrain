# Speech-to-Speech Translation (with LJSpeech)
This folder contains the recipes for training unit-based HiFi-GAN vocoder with the popular LJSpeech dataset.
To train the speech-to-unit translation system, please refer to the CVSS recipe.

## Dataset
The dataset can be downloaded from here:
https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

## Installing Extra Dependencies
Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:
```
pip install -r extra_requirements.txt
```

## Quantization
The first step is to train the quantization model to decode speech units from speech.
```
cd Quantization
python train.py --device=cuda:0 hparams/kmeans.yaml --data_folder=/your_folder/LJSpeech-1.1
```

## Unit HiFi-GAN
Next we can train the HiFi-GAN vocoder using previously trained k-means model.
```
python train.py --device=cuda:0 --max_grad_norm=1.0 hparams/train.yaml --kmeans_folder=./Quantization/results/kmeans/4321/save --data_folder=/your_folder/LJSpeech-1.1
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

