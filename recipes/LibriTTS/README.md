# Text-to-Speech with LibriTTS
This folder contains the recipes for training TTS systems (including vocoders) with the LibriTTS dataset.

# Dataset
The LibriTTS dataset is available here: https://www.openslr.org/60/, https://www.openslr.org/resources/60/

The `libritts_prepare.py` file automatically downloads the dataset if not present and has facilities to provide the names of the subsets to be downloaded.

# Zero-Shot Multi-Speaker Tacotron2
The subfolder "TTS/mstacotron2" contains the recipe for training a zero-shot multi-speaker version of the [Tacotron2](https://arxiv.org/abs/1712.05884) model.
To run this recipe, go into the `"TTS/mstacotron2"` folder and run:

```bash
python train.py hparams/train.yaml --data_folder=/path/to/libritts_data --device=cuda:0 --max_grad_norm=1.0
```

Please ensure that you use absolute paths when specifying the data folder.

Training time required on NVIDIA A100 GPU using LibriTTS train-clean-100 and train-clean-360 subsets: ~ 2 hours 54 minutes per epoch

The training logs are available [here](https://www.dropbox.com/sh/ti2vk7sce8f9fgd/AABcDGWCrBvLX_ZQs76mlJRYa?dl=0).

The pre-trained model with an easy-inference interface is available on [HuggingFace](https://huggingface.co/speechbrain/tts-mstacotron2-libritts).

**Please Note**: The current model effectively captures speaker identities. Nevertheless, the synthesized speech quality exhibits some metallic characteristics and may include artifacts like overly long pauses.
We are actively working to enhancing the model and will release updates as soon as improvements are achieved. We warmly welcome contributions from the community to collaboratively make the model even better!

# HiFi GAN (Vocoder)
The subfolder "vocoder/hifi_gan/" contains the [HiFi GAN vocoder](https://arxiv.org/pdf/2010.05646.pdf).
The vocoder is a neural network that converts a spectrogram into a waveform (it can be used on top of Tacotron2).

We suggest using `tensorboard_logger` by setting `use_tensorboard: True` in the yaml file. Thus, `Tensorboard` should be installed.

To run this recipe, go into the `"vocoder/hifigan/"` folder and run:

```bash
python train.py hparams/train.yaml --data_folder=/path/to/LibriTTS
```

The recipe will automatically download the LibriTTS dataset and resamples it as specified.

Training time required on NVIDIA A100 GPU using LibriTTS train-clean-100 and train-clean-360 subsets: ~ 1 hour 50 minutes per epoch

The training logs and checkpoints are available [here](https://www.dropbox.com/sh/gjs1kslxkxz819q/AABPriN4dOoD1qL7NoIyVk0Oa?dl=0).

To change the sample rate for model training go to the `"vocoder/hifigan/hparams/train.yaml"` file and change the value for `sample_rate` as required.

On HuggingFace, you can find the following pretrained models (with easy-inference interface):
- https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz
- https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz

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
