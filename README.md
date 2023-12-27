<p align="center">
  <img src="https://raw.githubusercontent.com/speechbrain/speechbrain/develop/docs/images/speechbrain-logo.svg" alt="SpeechBrain Logo"/>
</p>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=40&duration=7000&pause=1000&random=false&width=1200&height=100&lines=Simplify+Conversational+AI+Development)](https://git.io/typing-svg)


| **[Tutorials](https://speechbrain.github.io/tutorial_basics.html)** | **[Website](https://speechbrain.github.io/)** | **[Documentation](https://speechbrain.readthedocs.io/en/latest/index.html)** | **[Contributing](https://speechbrain.readthedocs.io/en/latest/contributing.html)** | **[HuggingFace](https://huggingface.co/speechbrain)** | **[YouTube](https://www.youtube.com/@SpeechBrainProject)** |


![GitHub Repo stars](https://img.shields.io/github/stars/speechbrain/speechbrain) 
<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> *Help our community project, star on GitHub!*

#
# 🗣️💬 What SpeechBrain Offers  

- SpeechBrain is an **open-source** [PyTorch](https://pytorch.org/) toolkit that accelerates **Conversational AI** development, i.e., the technology behind *speech assistants*, *chatbots*, and *large language models*. 

- It is crafted for fast and easy creation of advanced technologies for **Speech** and **Text** Processing.


## 🌐  Vision 
- With the rise of [deep learning](https://www.deeplearningbook.org/), once-distant domains like speech processing and NLP are now very close. A well-designed neural network and large datasets are all you need. 

- We think it is now time for a **holistic toolkit** that, mimicking the human brain, jointly supports diverse technologies for complex Conversational AI systems. 

- This spans *speech recognition*, *speaker recognition*, *speech enhancement*, *speech separation*, *language modeling*, *dialogue*, and beyond. 



## 📚 Training Recipes
- We share over 200 competitive training [recipes](https://github.com/speechbrain/speechbrain/tree/develop/recipes) on more than 40 datasets supporting 20 speech and text processing tasks (see below). 

- We support both training from scratch and fine-tuning pretrained models such as [Whisper](https://huggingface.co/openai/whisper-large), [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2), [WavLM](https://huggingface.co/docs/transformers/model_doc/wavlm), [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert), [GPT2](https://huggingface.co/gpt2), [Llama2](https://huggingface.co/docs/transformers/model_doc/llama2), and beyond. The models on [HuggingFace](https://huggingface.co/) can be easily plugged in and fine-tuned.

- For any task, you train the model using these commands:
```bash
python train.py hparams/train.yaml
```

- The hyperparameters are encapsulated in a YAML file, while the training process is orchestrated through a Python script. 

- We maintained a consistent code structure across different tasks. 

- For better replicability, training logs and checkpoints are hosted on Dropbox.

## <a href="https://huggingface.co/speechbrain" target="_blank"> <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> </a> Pretrained Models and Inference

- Access over 100 pretrained models hosted on [HuggingFace](https://huggingface.co/speechbrain).
- Each model comes with a user-friendly interface for seamless inference. For example, transcribing speech using a pretrained model requires just three lines of code:

```python
from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
asr_model.transcribe_file("speechbrain/asr-conformer-transformerlm-librispeech/example.wav")
```

##  <a href="https://speechbrain.github.io/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/1200px-Google_Colaboratory_SVG_Logo.svg.png" alt="drawing" width="50"/> </a>  Documentation
- We are deeply dedicated to promoting inclusivity and education.
- We have authored over 30 [tutorials](https://speechbrain.github.io/) on Google Colab that not only describe how SpeechBrain works but also help users familiarize themselves with Conversational AI.
- Every class or function has clear explanations and examples that you can run. Check out the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html) for more details 📚.



## 🎯 Use Cases
- 🚀 **Research Acceleration**: Speeding up academic and industrial research. You can develop and integrate new models effortlessly, comparing their performance against our baselines.
  
- ⚡️ **Rapid Prototyping**: Ideal for quick prototyping in time-sensitive industrial projects, embraced by large corporations and startups.
  
- 🎓 **Educational Tool**: SpeechBrain's simplicity makes it a valuable educational resource. It is used by institutions like [Mila](https://mila.quebec/en/), [Concordia University](https://www.concordia.ca/), [Avignon University](https://univ-avignon.fr/en/), and many others for student training.

#
# 🚀 Quick Start 

To get started with SpeechBrain, follow these simple steps:

## 🛠️ Installation 

### Install via PyPI

1. Install SpeechBrain using PyPI:

    ```bash
    pip install speechbrain
    ```

2. Access SpeechBrain in your Python code:

    ```python
    import speechbrain as sb
    ```

### Install from GitHub
This installation is recommended for users who wish to conduct experiments and customize the toolkit according to their needs.

1. Clone the GitHub repository and install the requirements:

    ```bash
    git clone https://github.com/speechbrain/speechbrain.git
    cd speechbrain
    pip install -r requirements.txt
    pip install --editable .
    ```

2. Access SpeechBrain in your Python code:

    ```python
    import speechbrain as sb
    ```

Any modifications made to the `speechbrain` package will be automatically reflected, thanks to the `--editable` flag.

## ✔️ Test Installation 

Ensure your installation is correct by running the following commands:

```bash
pytest tests
pytest --doctest-modules speechbrain
```

## 🏃‍♂️ Running an Experiment 

In SpeechBrain, you can train a model for any task using the following steps:

```bash
cd recipes/<dataset>/<task>/
python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the YAML file. 

# 🔧 Supported Technologies
- SpeechBrain is a versatile framework designed for implementing a wide range of technologies within the field of Conversational AI.
- It excels not only in individual task implementations but also in seamlessly combining various technologies into complex pipelines.
  
## Speech/Audio Processing
| Tasks        | Datasets           | Technologies  |
| ------------- |-------------| -----|
| Speech Recognition      | [AISHELL-1](https://github.com/speechbrain/speechbrain/tree/develop/recipes/AISHELL-1), [CommonVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice), [DVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/DVoice), [KsponSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/KsponSpeech), [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech), [MEDIA](https://github.com/speechbrain/speechbrain/tree/develop/recipes/MEDIA), [RescueSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/RescueSpeech), [Switchboard](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Switchboard), [TIMIT](https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT), [Tedlium2](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Tedlium2), [Voicebank](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank) | CTC, Tranducers, Tranformers, Seq2Seq, Beamsearch, Rescoring, Conformer,Streamable Conformer, Branchformer, Hyperconformer |
| Speaker Recognition      | [VoxCeleb](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb) | ECAPA-TDNN, ResNET, Xvectors, PLDA, Score Normalization |
| Speech Separation      | [WSJ0Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WSJ0Mix), [LibriMix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriMix), [WHAM!](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WHAMandWHAMR), [WHAMR!](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WHAMandWHAMR), [Aishell1Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Aishell1Mix), [BinauralWSJ0Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/BinauralWSJ0Mix) | SepFormer, RESepFormer, SkiM, DualPath RNN, ConvTasNET |
| Speech Enhancement      | [DNS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/DNS), [Voicebank](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank) | SepFormer, MetricGAN, MetricGAN-U, SEGAN, spectral masking, time masking |
| Text-to-Speech      | [LJSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech), [LibriTTS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriTTS) | Tacotron2, Multi-Speaker Tacotron2, FastSpeech2 |
| Vocoding      | [LJSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech), [LibriTTS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriTTS) | HiFiGAN, DiffWave |
| Spoken Language Understanding | [MEDIA](https://github.com/speechbrain/speechbrain/tree/develop/recipes/MEDIA), [SLURP](https://github.com/speechbrain/speechbrain/tree/develop/recipes/SLURP), [Fluent Speech Commands](https://github.com/speechbrain/speechbrain/tree/develop/recipes/fluent-speech-commands), [Timers-and-Such](https://github.com/speechbrain/speechbrain/tree/develop/recipes/timers-and-such)  | Direct SLU, Decoupled SLU, Multistage SLU |
| Speech-to-Speech Translation  | [CVSS](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CVSS) | Discrete Hubert, HifiGAN, wav2vec2 |
| Speech Translation  | [Fisher CallHome (Spanish)](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Fisher-Callhome-Spanish), [IWSLT22(lowresource)](https://github.com/speechbrain/speechbrain/tree/develop/recipes/IWSLT22_lowresource) | wav2vec2 |
| Emotion Classification      | [IEMOCAP](https://github.com/speechbrain/speechbrain/tree/develop/recipes/IEMOCAP), [ZaionEmotionDataset](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ZaionEmotionDataset) | ECAPA-TDNN, Wav2vec2, Emotion Diarization |
| Language Identification | [VoxLingua107](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107), [CommonLanguage](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonLanguage)| ECAPA-TDNN |
| Voice Activity Detection  | [LibriParty](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriParty) | CRDNN |
| Sound Classification  | [ESC50](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ESC50), [UrbanSound](https://github.com/speechbrain/speechbrain/tree/develop/recipes/UrbanSound8k) | CNN, CNN14, ECAPA-TDNN |
| Self-Supervised Learning | [CommonVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice), [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech) | Wav2vec2 |
| Interpretabiliy | [ESC50](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ESC50) | Learning-to-Interpret (L2I), Non-Negative Matrix Factorization (NMF), PIQ |
| Speech Generation | [AudioMNIST](https://github.com/speechbrain/speechbrain/tree/develop/recipes/AudioMNIST) | Diffusion, Latent Diffusion |
| Metric Learning | [REAL-M](https://github.com/speechbrain/speechbrain/tree/develop/recipes/REAL-M/sisnr-estimation)| Blind SNR-Estimation |
| Allignment | [TIMIT](https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT) | CTC, Viterbi, Forward Forward |
| Diarization | [AMI](https://github.com/speechbrain/speechbrain/tree/develop/recipes/AMI) | ECAPA-TDNN, X-vectors, Spectral Clustering |

## Text Processing
| Tasks        | Datasets           | Technologies  |
| ------------- |-------------| -----|
| Language Modeling | [CommonVoice](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice), [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/unstable-v0.6/recipes/LibriSpeech)| n-grams, RNNLM, TransformerLM |
| Response Generation | [MultiWOZ](https://github.com/speechbrain/speechbrain/tree/unstable-v0.6/recipes/MultiWOZ/response_generation)| GPT2, Llama2 |
| Grapheme-to-Phoneme | [LibriSpeech](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech) | RNN, Transformer, Curriculum Learning, Homograph loss |


SpeechBrain is rapidly evolving, with ongoing efforts to support a growing array of technologies in the future.

# 📊 Performance 

- SpeechBrain integrates a variety of technologies, including those that achieves competitive or state-of-the-art performance.

- For a comprehensive overview of the achieved performance across different tasks, datasets, and technologies, please visit [here](https://github.com/speechbrain/speechbrain/blob/develop/PERFORMANCE.md).



# Key features

SpeechBrain provides various useful tools to speed up and facilitate research on speech and language technologies:
- Various pretrained models nicely integrated with <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> <sub>(HuggingFace)</sub> in our official [organization account](https://huggingface.co/speechbrain). These models are coupled with easy-inference interfaces that facilitate their use.  To help everyone replicate our results, we also provide all the experimental results and folders (including logs, training curves, etc.) in a shared Google Drive folder.
- The `Brain` class is a fully-customizable tool for managing training and evaluation loops over data. The annoying details of training loops are handled for you while retaining complete flexibility to override any part of the process when needed.
- A YAML-based hyperparameter file that specifies all the hyperparameters, from individual numbers (e.g., learning rate) to complete objects (e.g., custom models). This elegant solution dramatically simplifies the training script.
- Multi-GPU training and inference with PyTorch Data-Parallel or Distributed Data-Parallel.
- Mixed-precision for faster training.
- A transparent and entirely customizable data input and output pipeline. SpeechBrain follows the PyTorch data loading style and enables users to customize the I/O pipelines (e.g., adding on-the-fly downsampling, BPE tokenization, sorting, threshold ...).
- On-the-fly dynamic batching
- Efficient reading of large datasets from a shared  Network File System (NFS) via [WebDataset](https://github.com/webdataset/webdataset).
- Interface with [HuggingFace](https://huggingface.co/speechbrain) for popular models such as wav2vec2  and Hubert.
- Interface with [Orion](https://github.com/Epistimio/orion) for hyperparameter tuning.


### Speech recognition

SpeechBrain supports state-of-the-art methods for end-to-end speech recognition:
- Support of wav2vec 2.0 pretrained model with finetuning.
- State-of-the-art performance or comparable with other existing toolkits in several ASR benchmarks.
- Easily customizable neural language models, including RNNLM and TransformerLM. We also share several pre-trained models that you can easily use (more to come!). We support the Hugging Face `dataset` to facilitate the training over a large text dataset.
- Hybrid CTC/Attention end-to-end ASR:
    - Many available encoders: CRDNN (VGG + {LSTM,GRU,Li-GRU} + DNN), ResNet, SincNet, vanilla transformers, whisper, context net-based transformers or conformers. Thanks to the flexibility of SpeechBrain, any fully customized encoder could be connected to the CTC/attention decoder and trained in a few hours of work. The decoder is fully customizable: LSTM, GRU, LiGRU, transformer, or your neural network!
    - Optimised and fast beam search on both CPUs and GPUs.
- Transducer end-to-end ASR with both a custom Numba loss and the torchaudio one. Any encoder or decoder can be plugged into the transducer ranging from VGG+RNN+DNN to conformers.
- Pre-trained ASR models for transcribing an audio file or extracting features for a downstream task.
- Fully customizable with the possibility to add external Beam Search decoders, if the ones offered natively by SpeechBrain are not sufficient, such as [PyCTCDecode](https://github.com/kensho-technologies/pyctcdecode) like in our LibriSpeech CTC wav2vec recipe.

### Feature extraction and augmentation

SpeechBrain provides efficient (GPU-friendly) speech augmentation and feature extraction pipelines:
- On-the-fly and fully-differentiable acoustic feature extraction: filter banks can be learned. This strategy simplifies the training pipeline (you don't have to dump features on disk).
- On-the-fly feature normalization (global, sentence, batch, or speaker level).
- On-the-fly environmental corruptions based on noise, reverberation, and babble for robust model training.
- On-the-fly frequency and time domain SpecAugment with speed augmentation.
- We support both SinConv and LEAF convolutional frontends.

### Speech enhancement and separation
- Recipes for spectral masking, spectral mapping, and time-domain speech enhancement.
- Multiple sophisticated enhancement losses, including differentiable STOI loss, MetricGAN, and mimic loss.
- State-of-the-art performance on speech separation with Conv-TasNet, DualPath RNN, SepFormer, and RE-SepFormer.

### Speaker recognition, identification and diarization
SpeechBrain provides different models for speaker recognition, identification, and diarization on different datasets:
- State-of-the-art performance on speaker recognition and diarization based on ECAPA-TDNN models.
- Original Xvectors implementation (inspired by Kaldi) with PLDA.
- Spectral clustering for speaker diarization (combined with speakers embeddings).
- Libraries to extract speaker embeddings with a pre-trained model on your data.

### Text-to-Speech (TTS) and Vocoders
- Recipes for training TTS systems such as [Tacotron2](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech/) and [FastSpeech2](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech/) with LJSpeech.
- Recipes for training Vocoders such as [HiFIGAN](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LJSpeech).

### Grapheme-to-Phoneme (G2P)
We have models for converting characters into a sequence of phonemes. In particular, we have Transformer- and RNN-based models operating at the sentence level (i.e, converting a full sentence into a corresponding sequence of phonemes). The models are trained with both data from Wikipedia and LibriSpeech.

### Language Identification
SpeechBrain provides different models for language identification.
In particular, our best model is based on an ECAPA-TDNN trained with the [voxlingua107 dataset](http://bark.phon.ioc.ee/voxlingua107/).

### Speech Translation
- Recipes for transformer and conformer-based end-to-end speech translation.
- Possibility to choose between normal training (Attention), multi-objectives (CTC+Attention), and multitasks (ST + ASR).

### Self-Supervised Learning of Speech Representations
- Recipes for wav2vec 2.0 pre-training with multiple GPUs compatible with HuggingFace models.

### Multi-microphone processing
Combining multiple microphones is a powerful approach to achieving robustness in adverse acoustic environments:
- Delay-and-sum, MVDR, and GeV beamforming.
- Speaker localization.

### Emotion Recognition
- Recipes for emotion recognition using SSL and ECAPA-TDNN models on the [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) dataset.
- Recipe for emotion diarization using SSL models on the [ZaionEmotionDataset](https://zaion.ai/en/resources/zaion-lab-blog/zaion-emotion-dataset/).

### Interpretability
- Recipes for various intepretability techniques on the ESC50 dataset.

### Spoken Language Understanding
- Recipes for training wav2vec 2.0 models on, [SLURP](https://zenodo.org/record/4274930#.YEFCYHVKg5k), [MEDIA](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/) and [timers-and-such](https://zenodo.org/record/4623772#.YGeMMHVKg5k) datasets.

### Performance
The recipes released with speechbrain implement speech processing systems with competitive or state-of-the-art performance. In the following, we report the best performance achieved on some popular benchmarks:

| Dataset        | Task           | System  | Performance  |
| ------------- |:-------------:| -----:|-----:|
| LibriSpeech      | Speech Recognition | wav2vec2 | WER=1.90% (test-clean) |
| LibriSpeech      | Speech Recognition | CNN + Conformer | WER=2.0% (test-clean) |
| TIMIT      | Speech Recognition | CRDNN + distillation | PER=13.1% (test) |
| TIMIT      | Speech Recognition | wav2vec2 + CTC/Att. | PER=8.04% (test) |
| CommonVoice (English) | Speech Recognition | wav2vec2 + CTC | WER=15.69% (test) |
| CommonVoice (French) | Speech Recognition | wav2vec2 + CTC | WER=9.96% (test) |
| CommonVoice (Italian) | Speech Recognition | wav2vec2 + seq2seq | WER=9.86% (test) |
| CommonVoice (Kinyarwanda) | Speech Recognition | wav2vec2 + seq2seq | WER=18.91% (test) |
| AISHELL (Mandarin) | Speech Recognition | wav2vec2 + CTC | CER=5.06% (test) |
| Fisher-callhome (spanish) | Speech translation | conformer (ST + ASR) | BLEU=48.04 (test) |
| VoxCeleb2      | Speaker Verification | ECAPA-TDNN | EER=0.80% (vox1-test) |
| AMI      | Speaker Diarization | ECAPA-TDNN | DER=3.01% (eval)|
| VoiceBank      | Speech Enhancement | MetricGAN+| PESQ=3.08 (test)|
| WSJ2MIX      | Speech Separation | SepFormer| SDRi=22.6 dB (test)|
| WSJ3MIX      | Speech Separation | SepFormer| SDRi=20.0 dB (test)|
| WHAM!     | Speech Separation | SepFormer| SDRi= 16.4 dB (test)|
| WHAMR!     | Speech Separation | SepFormer| SDRi= 14.0 dB (test)|
| Libri2Mix     | Speech Separation | SepFormer| SDRi= 20.6 dB (test-clean)|
| Libri3Mix     | Speech Separation | SepFormer| SDRi= 18.7 dB (test-clean)|
| LibryParty | Voice Activity Detection | CRDNN | F-score=0.9477 (test) |
| IEMOCAP | Emotion Recognition | wav2vec2 | Accuracy=79.8% (test) |
| CommonLanguage | Language Recognition | ECAPA-TDNN | Accuracy=84.9% (test) |
| Timers and Such | Spoken Language Understanding | CRDNN | Intent Accuracy=89.2% (test) |
| SLURP | Spoken Language Understanding | HuBERT | Intent Accuracy=87.54% (test) |
| VoxLingua 107 | Identification | ECAPA-TDNN | Sentence Accuracy=93.3% (test) |

For more details, take a look at the corresponding implementation in recipes/dataset/.

### Pretrained Models

Beyond providing recipes for training the models from scratch, SpeechBrain shares several pre-trained models (coupled with easy-inference functions) on [HuggingFace](https://huggingface.co/speechbrain). In the following, we report some of them:

| Task        | Dataset | Model |
| ------------- |:-------------:| -----:|
| Speech Recognition | LibriSpeech | [CNN + Transformer](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) |
| Speech Recognition | LibriSpeech | [CRDNN](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) |
| Speech Recognition | CommonVoice(English) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-en) |
| Speech Recognition | CommonVoice(French) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-fr) |
| Speech Recognition | CommonVoice(Italian) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-it) |
| Speech Recognition | CommonVoice(Kinyarwanda) | [wav2vec + CTC](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-rw) |
| Speech Recognition | AISHELL(Mandarin) | [wav2vec + seq2seq](https://huggingface.co/speechbrain/asr-wav2vec2-transformer-aishell) |
| Text-to-Speech | LJSpeech | [Tacotron2](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech) |
| Speaker Recognition | Voxceleb | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| Speech Separation | WHAMR! | [SepFormer](https://huggingface.co/speechbrain/sepformer-whamr) |
| Speech Enhancement | Voicebank | [MetricGAN+](https://huggingface.co/speechbrain/metricgan-plus-voicebank) |
| Speech Enhancement | WHAMR! | [SepFormer](https://huggingface.co/speechbrain/sepformer-whamr-enhancement) |
| Spoken Language Understanding | Timers and Such | [CRDNN](https://huggingface.co/speechbrain/slu-timers-and-such-direct-librispeech-asr) |
| Language Identification | CommonLanguage | [ECAPA-TDNN](https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa) |

The full list of pre-trained models can be found on [HuggingFace](https://huggingface.co/speechbrain)

### Documentation & Tutorials
SpeechBrain is designed to speed up the research and development of speech technologies. Hence, our code is backed-up with different levels of documentation:
- **Educational-level:** we provide various Google Colab (i.e., interactive) tutorials describing all the building blocks of SpeechBrain ranging from the core of the toolkit to a specific model designed for a particular task. The tutorials are designed not only to help people familiarize themselves with SpeechBrain but, more in general, to help them familiarize themselves with speech and language technologies.
- **Functional-level:** all classes in SpeechBrain contains a detailed docstring. It describes the input and output formats, the different arguments, the usage of the function, the potentially associated bibliography, and a function example used for test integration during pull requests.
- **Low-level:** The code also uses a lot of in-line comments to describe nontrivial parts of the code.

### Under development
We are currently implementing speech synthesis pipelines and real-time speech processing pipelines. An interface with the Finite State Transducers (FST) implemented by the [Kaldi 2 team](https://github.com/k2-fsa/k2) is under development.

# Where is what, a link list.
```
                  (documentation)           (tutorials)
                  .—————————————.            .———————.
                  | readthedocs |       ‚––> | Colab |
                  \—————————————/      ∕     \———————/
                         ^       ‚––––‘          |
    (release)            |      ∕                v
    .——————.       .———————————. (landing) .———————————.
    | PyPI | –––>  | github.io |  (page)   | templates |   (reference)
    \——————/       \———————————/       ‚–> \———————————/ (implementation)
        |                |        ‚–––‘          |
        v                v       ∕               v
.———————————–—.   .———————————–—.           .—————————.           .~~~~~~~~~~~~~.
| HyperPyYAML |~~~| speechbrain | ––––––––> | recipes | ––––––––> | HuggingFace |
\————————————–/   \————————————–/           \—————————/     ∕     \~~~~~~~~~~~~~/
  (usability)     (source/modules)          (use cases)    ∕    (pretrained models)
                                                          ∕
                        |                        |       ∕               |
                        v                        v      ∕                v
                  .~~~~~~~~~~~~~.            .~~~~~~~~.            .———————————.
                  |   PyTorch   | ––––––––-> | GDrive |            | Inference |
                  \~~~~~~~~~~~~~/            \~~~~~~~~/            \———————————/
                   (checkpoints)             (results)            (code snippets)
```

* https://speechbrain.github.io/
  * via: https://github.com/speechbrain/speechbrain.github.io
  * pointing to several tutorials on Google Colab
* https://github.com/speechbrain/speechbrain
  * [docs](https://github.com/speechbrain/speechbrain/tree/develop/docs) for https://speechbrain.readthedocs.io/
  * [recipes](https://github.com/speechbrain/speechbrain/tree/develop/recipes)
  * [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain), heavily tied with [HyperPyYAML](https://github.com/speechbrain/HyperPyYAML); released on [PyPI](https://pypi.org/project/speechbrain/)
  * [templates](https://github.com/speechbrain/speechbrain/tree/develop/templates)
  * [tools](https://github.com/speechbrain/speechbrain/tree/develop/tools) for non-core functionality
* https://huggingface.co/speechbrain/
  * hosting several model cards (pretrained models with code snippets)
* Gdrive
  * hosting training results; checkpoints; ...

# Conference Tutorials
SpeechBrain has been presented at Interspeech 2021 and 2022 as well as ASRU 2021. When possible, we will provide some ressources here:
- [Interspeech 2022 slides.](https://drive.google.com/drive/folders/1d6GAquxw6rZBI-7JvfUQ_-upeiKstJEo)
- [Interspeech 2021 YouTube recordings.](https://www.youtube.com/results?search_query=Interspeech+speechbrain+)
- [SpeechBrain 2023 Online Summit YouTube recordings.](https://www.youtube.com/playlist?list=PL9PacmG8A8SfRDaeVcnrlY7eoEMYGANLB)

# Quick installation
SpeechBrain is constantly evolving. New features, tutorials, and documentation will appear over time.
SpeechBrain can be installed via PyPI. Moreover,  a local installation can be used by those users who want to run experiments and modify/customize the toolkit. SpeechBrain supports both CPU and GPU computations. For most all the recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.


## Install via PyPI

Once you have created your Python environment (Python 3.7+) you can simply type:

```
pip install speechbrain
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

## Install with GitHub

Once you have created your Python environment (Python 3.7+) you can simply type:

```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

Any modification made to the `speechbrain` package will be automatically interpreted as we installed it with the `--editable` flag.

## Test Installation
Please, run the following script to make sure your installation is working:
```
pytest tests
pytest --doctest-modules speechbrain
```

# Running an experiment
In SpeechBrain, you can run experiments in this way:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the yaml file. The folder is created by calling `sb.core.create_experiment_directory()` in `experiment.py`. Both detailed logs and experiment outputs are saved there. Furthermore, less verbose logs are output to stdout.

# SpeechBrain Roadmap

As a community-based and open-source project, SpeechBrain needs the help of its community to grow in the right direction. Opening the roadmap to our users enables the toolkit to benefit from new ideas, new research axes, or even new technologies. The roadmap will be available in our [GitHub Discussions](https://github.com/speechbrain/speechbrain/discussions/categories/announcements) and will list all the changes and updates that need to be done in the current version of SpeechBrain. Users are more than welcome to propose new items via new Discussions topics!

# Learning SpeechBrain

We provide users with different resources to learn how to use SpeechBrain:
- General information can be found on the [website](https://speechbrain.github.io).
- We offer many tutorials, you can start from the [basic ones](https://speechbrain.github.io/tutorial_basics.html) about SpeechBrain's basic functionalities and building blocks. We provide also more advanced tutorials (e.g SpeechBrain advanced, signal processing ...). You can browse them via the Tutorials drop-down menu on [SpeechBrain website](https://speechbrain.github.io) in the upper right.
- Details on the SpeechBrain API, how to contribute, and the code are given in the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html).

# License
SpeechBrain is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Note that this project has no connection to the Apache Foundation, other than that we use the same license terms.

# Social Media
We constantly update the community using Twitter. [Feel free to follow us](https://twitter.com/speechbrain1)

# Citing SpeechBrain
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

