# Text-to-Speech (with LJSpeech)
This folder contains the recipes for training TTS systems (including vocoders) wiith the popular LJSpeech dataset.

# Dataset
The dataset can be downloaded from here:
https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# Tacotron 2
The subfolder "tacotron2" contains the recipe for training the popular [tacotron2](https://arxiv.org/abs/1712.05884) TTS model.
To run this recipe, go into the "tacotron2" folder and run:

```
python train.py --device=cuda:0 --max_grad_norm=1.0 --data_folder=/your_folder/LJSpeech-1.1 hparams/train.yaml
```

The training logs are available [here](https://www.dropbox.com/sh/1npvo1g1ncafipf/AAC5DR1ErF2Q9V4bd1DHqX43a?dl=0).

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech).

# FastSpeech2
The subfolder "fastspeech2" contains the recipe for training the non-autoregressive transformer based TTS model [FastSpeech2](https://arxiv.org/abs/2006.04558).

Training FastSpeech2 requires pre-extracted phoneme alignments (durations). The LJSpeech phoneme alignments from Montreal Forced Aligner can be automatically downloaded, decompressed and stored at this location: ```/your_folder/LJSpeech-1.1/TextGrid```.

To run this recipe, go into the "fastspeech2" folder and run:

```
pip install -r extra_requirements.txt

python train.py --data_folder=/your_folder/LJSpeech-1.1 hparams/train.yaml
```
Training takes about 3 minutes/epoch on 1 * V100 32G.

The training logs are available [here](https://www.dropbox.com/sh/tqyp58ogejqfres/AAAtmq7cRoOR3XTsq0iSgyKBa?dl=0).

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-fastspeech2-ljspeech).

# HiFi GAN (Vocoder)
The subfolder "vocoder/hifi_gan/" contains the [HiFi GAN vocoder](https://arxiv.org/pdf/2010.05646.pdf).
The vocoder is a neural network that converts a spectrogram into a waveform (it can be used on top of Tacotron 2).

We suggest using `tensorboard_logger` by setting `use_tensorboard: True` in the yaml file, thus `Tensorboard` should be installed.

To run this recipe, go into the "vocoder/hifi_gan/" folder and run:

```
python train.py hparams/train.yaml --data_folder /path/to/LJspeech
```

The training logs are available [here](https://drive.google.com/drive/folders/19sLwV7nAsnUuLkoTu5vafURA9Fo2WZgG?usp=sharing)

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-hifigan-ljspeech).


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

