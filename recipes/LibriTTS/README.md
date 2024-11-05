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

For now, enhancements are needed for training the model from scratch when train-clean-360 is included. Inference can be effectuated with `clone_voice_char_input` function in the MSTacotron2 interface.

The pre-trained model (a model fine-tuned from LJSpeech tacotron2) with an easy-inference interface is available on [HuggingFace](https://huggingface.co/speechbrain/tts-mstacotron2-libritts).

**Please Note**: The current model effectively captures speaker identities. Nevertheless, the synthesized speech quality exhibits some metallic characteristics and may include artifacts like overly long pauses.
We are actively working to enhancing the model and will release updates as soon as improvements are achieved. We warmly welcome contributions from the community to collaboratively make the model even better!

# HiFi GAN (Vocoder)
The subfolder "vocoder/hifigan/" contains the [HiFi GAN vocoder](https://arxiv.org/pdf/2010.05646.pdf).
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

# HiFiGAN Unit Vocoder
The subfolder "vocoder/hifigan_discrete/" contains the [HiFiGAN Unit vocoder](https://arxiv.org/abs/2406.10735). This vocoder is a neural network designed to transform discrete self-supervised representations into waveform data.
This is suitable for a wide range of generative tasks such as speech enhancement, separation, text-to-speech, voice cloning, etc. Please read [DASB - Discrete Audio and Speech Benchmark](https://arxiv.org/abs/2406.14294) for more information.

To run this recipe successfully, start by installing the necessary extra dependencies:

```bash
pip install -r extra_requirements.txt
```

Before training the vocoder, you need to choose a speech encoder to extract representations that will be used as discrete audio input. We support k-means models using features from HuBERT, WavLM, or Wav2Vec2. Below are the available self-supervised speech encoders for which we provide pre-trained k-means checkpoints:

| Encoder  | HF model                                |
|----------|-----------------------------------------|
| HuBERT   | facebook/hubert-large-ll60k             |
| Wav2Vec2 | facebook/wav2vec2-large-960h-lv60-self  |
| WavLM    | microsoft/wavlm-large                   |

Checkpoints are available in the HF [SSL_Quantization](https://huggingface.co/speechbrain/SSL_Quantization) repository. Alternatively, you can train your own k-means model by following instructions in the "LJSpeech/quantization" README.

Next, configure the SSL model type, k-means model, and corresponding hub in your YAML configuration file. Follow these steps:

1. Navigate to the "vocoder/hifigan_discrete/hparams" folder and open "train.yaml" file.
2. Modify the `encoder_type` field to specify one of the SSL models: "HuBERT", "WavLM", or "Wav2Vec2".
3. Update the `encoder_hub` field with the specific name of the SSL Hub associated with your chosen model type.

If you have trained your own k-means model, follow these additional steps:

4. Update the `kmeans_folder` field with the specific name of the SSL Hub containing your trained k-means model. Please follow the same file structure as the official one in [SSL_Quantization](https://huggingface.co/speechbrain/SSL_Quantization).
5. Update the `kmeans_dataset` field with the specific name of the dataset on which the k-means model was trained.
6. Update the `num_clusters` field according to the number of clusters of your k-means model.

Finally, navigate back to the "vocoder/hifigan_discrete/" folder and run the following command:

```bash
python train.py hparams/train.yaml --data_folder=/path/to/LibriTTS
```

Additionally, we provide support for external speaker embeddings along with discrete tokens. By default, the speaker model used is ECAPA-TDNN trained on the VoxCeleb dataset. For more information, you can find the pretrained model on [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec).
To run it, use the following command:

```bash
python train.py hparams/train_spk.yaml --data_folder=/path/to/LibriTTS
```

Training typically takes around 15 minutes per epoch when using an NVIDIA A100 40G.

On HuggingFace, you can find the following pretrained models (with easy-inference interface):
- https://huggingface.co/speechbrain/hifigan-hubert-l1-3-7-12-18-23-k1000-LibriTTS
- https://huggingface.co/speechbrain/hifigan-wav2vec-l1-3-7-12-18-23-k1000-LibriTTS
- https://huggingface.co/speechbrain/hifigan-wavlm-l1-3-7-12-18-23-k1000-LibriTTS

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
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
