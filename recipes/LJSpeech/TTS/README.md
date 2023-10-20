# Text-to-Speech (with LJSpeech)
This folder contains the recipes for training TTS systems (including vocoders) wiith the popular LJSpeech dataset.

# Dataset
The dataset can be downloaded from here:
https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# Tacotron 2
The subfolder "tacotron2" contains the recipe for training the popular [tacotron2](https://arxiv.org/abs/1712.05884) TTS model.
To run this recipe, go into the "tacotron2" folder and run:

```
python train.py --device=cuda:0 --max_grad_norm=1.0 --data_folder=/your_folder/LJSpeech-1.1 hparams/train.yaml
```

The training logs are available [here](https://www.dropbox.com/sh/1npvo1g1ncafipf/AAC5DR1ErF2Q9V4bd1DHqX43a?dl=0).

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech).

# FastSpeech2
The subfolder "fastspeech2" contains the recipes for training the non-autoregressive transformer based TTS model [FastSpeech2](https://arxiv.org/abs/2006.04558).

### FastSpeech2 with pre-extracted durations from a forced aligner
Training FastSpeech2 requires pre-extracted phoneme alignments (durations). The LJSpeech phoneme alignments from Montreal Forced Aligner are automatically downloaded, decompressed and stored at this location: ```/your_folder/LJSpeech-1.1/TextGrid```.

To run this recipe, please first install the extra-dependencies :

```
pip install -r extra_requirements.txt
````

Then go into the "fastspeech2" folder and run:

```
python train.py --data_folder=/your_folder/LJSpeech-1.1 hparams/train.yaml
```
Training takes about 3 minutes/epoch on 1 * V100 32G.

The training logs are available [here](https://www.dropbox.com/sh/5xqq5tafeofv2y0/AABrHWGpjFr_RxDPVaVQUrSWa?dl=0).

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-fastspeech2-ljspeech).

### FastSpeech2 with internal alignment
This recipe allows training FastSpeech2 without forced aligner referring to [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf). The alignment can be learnt by an internal alignment network that is added to FastSpeech2. This recipe aims to simplify training when using custom data and provide better alignments for punctuations.

To run this recipe, please first install the extra-requirements:
```
pip install -r extra_requirements.txt
```
Then go into the "fastspeech2" folder and run:
```
python train_internal_alignment.py hparams/train_internal_alignment.yaml --data_folder=/your_folder/LJSpeech-1.1
```
The data preparation includes a grapheme-to-phoneme process for the entire corpus which may take several hours. Training takes about 5 minutes/epoch on 1 * V100 32G.

The training logs are available [here](https://www.dropbox.com/sh/p6hjkequkb0acik/AABo6YVAlzCjM_KS7-za0hhia?dl=0).

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-fastspeech2-internal-alignment-ljspeech).

# HiFiGAN (Vocoder)
The subfolder "vocoder/hifi_gan/" contains the [HiFiGAN vocoder](https://arxiv.org/pdf/2010.05646.pdf).
The vocoder is a neural network that converts a spectrogram into a waveform (it can be used on top of Tacotron2/FastSpeech2).

We suggest using `tensorboard_logger` by setting `use_tensorboard: True` in the yaml file, thus `Tensorboard` should be installed.

To run this recipe, go into the "vocoder/hifi_gan/" folder and run:

```
python train.py hparams/train.yaml --data_folder /path/to/LJspeech
```

Training takes about 10 minutes/epoch on an nvidia RTX8000.

The training logs are available [here](https://www.dropbox.com/sh/m2xrdssiroipn8g/AAD-TqPYLrSg6eNxUkcImeg4a?dl=0)

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-hifigan-ljspeech).

# DiffWave (Vocoder)
The subfolder "vocoder/diffwave/" contains the [Diffwave](https://arxiv.org/pdf/2009.09761.pdf) vocoder.

DiffWave is a versatile diffusion model for audio synthesis, which produces high-fidelity audio in different waveform generation tasks, including neural vocoding conditioned on mel spectrogram, class-conditional generation, and unconditional generation.

Here it serves as a vocoder that generates waveforms given spectrograms as conditions (it can be used on top of Tacotron2/FastSpeech2).

To run this recipe, go into the "vocoder/diffwave/" folder and run:

```
python train.py hparams/train.yaml --data_folder /path/to/LJspeech
```

The scripts will output a synthesized audio to `<output_folder>/samples` for a certain interval of training epoch.

We suggest using tensorboard_logger by setting `use_tensorboard: True` in the yaml file, thus Tensorboard should be installed.

Training takes about 6 minutes/epoch on 1 * V100 32G.

The training logs are available [here](https://www.dropbox.com/sh/tbhpn1xirtaix68/AACvYaVDiUGAKURf2o-fvgMoa?dl=0)

For inference, by setting `fast_sampling: True` , a fast sampling can be realized by passing user-defined variance schedules. According to the paper, high-quality audios can be generated with only 6 steps. This is highly recommanded.

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-diffwave-ljspeech).

# K-means (Quantization)
The subfolder "quantization" contains K-means clustering model. The model serves to quantize self-supervised representations into discrete representation. Thus representations can be used as target for speech-to-speech translation or as input for HiFiGAN Unit. By default, we use the 6th layer of HuBERT and set `k=100`.

To run this recipe, please first install the extra-dependencies :
```
pip install -r extra_requirements.txt
```
Then go into the "quantization" folder and run:
```
python train.py hparams/kmeans.yaml --data_folder=/path/to/LJspeech
```

# HiFiGAN Unit Vocoder
The subfolder "vocoder/hifi_gan_unit/" contains the [HiFiGAN Unit vocoder](https://arxiv.org/abs/2104.00355). This vocoder is a neural network designed to transform discrete self-supervised representations into waveform data and is suitable for speech-to-speech translation on top of CVSS/S2ST models. The discrete representations required by the vocoder are learned using k-means quantization, as previously described. Please ensure that you have executed the quantization step before proceeding with this script.

To run this recipe successfully, start by installing the necessary extra dependencies:

```bash
pip install -r extra_requirements.txt
```

Then, navigate to the "vocoder/hifi_gan_unit/" folder and run the following command:

```bash
python train.py hparams/train.yaml --kmeans_folder=/path/to/Kmeans/ckpt --data_folder=/path/to/LJspeech
```

The `kmeans_folder` should be specified based on the results of the previous quantization step (e.g., ../../quantization/results/kmeans/4321/save).

Training typically takes around 4 minutes per epoch when using an NVIDIA A100 40G.

You can access the pre-trained model, along with an easy-to-use inference function, on [HuggingFace](https://huggingface.co/speechbrain/tts-hifigan-unit-hubert-l6-k100-ljspeech).


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

