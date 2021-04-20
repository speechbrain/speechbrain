# The SpeechBrain Toolkit

<p align="center">
  <img src="http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/img/logo_noname_rounded_big.png" alt="drawing" width="250"/>
</p>

SpeechBrain is an **open-source** and **all-in-one** speech toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **multi-microphone signal processing** and many others.

*SpeechBrain is currently in beta*.

| **[Discourse](https://speechbrain.discourse.group)** | **[Tutorials](https://speechbrain.github.io/tutorial_basics.html)** | **[Website](https://speechbrain.github.io/)** | **[Documentation](https://speechbrain.readthedocs.io/en/latest/index.html)** | **[Contributing](https://speechbrain.readthedocs.io/en/latest/contributing.html)** | **[HuggingFace](https://huggingface.co/speechbrain)** |

# Key features

SpeechBrain provides various useful tools to speed up and facilitate research on speech technologies:
- Various pretrained models nicely integrated with <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="drawing" width="40"/> <sub>(HuggingFace)</sub> in our official [organization account](https://huggingface.co/speechbrain). These models are given with an interface to easily run inference, facilitating integration. If a *HuggingFace* model isn't available, we usually provide a least a Google Drive folder containing all the experimental results corresponding.
- The `Brain` class, a fully-customizable tool for managing training and evaluation loops over data. The annoying details of training loops are handled for you while retaining complete flexibility to override any part of the process when needed.
- A YAML-based hyperparameter specification language that describes all types of hyperparameters, from individual numbers (e.g. learning rate) to complete objects (e.g. custom models). This dramatically simplifies recipe code by distilling basic algorithmic components.
- Multi-GPU training and inference with PyTorch Data-Parallel or Distributed Data-Parallel.
- Mixed-precision for faster training.
- A transparent and entirely customizable data input and output pipeline. SpeechBrain follows the PyTorch data loader and dataset style and enables users to customize the i/o pipelines (e.g adding on-the-fly downsampling, BPE tokenization, sorting, threshold ...).


### Speech recognition

SpeechBrain supports state-of-the-art methods for end-to-end speech recognition:
- State-of-the-art performance or comparable with other existing toolkits in several ASR benchmarks.
- Easily customizable neural language models including RNNLM and TransformerLM. We also propose few pre-trained models to save you computations (more to come!). We support the Hugging Face `dataset` to facilitate the training over a large text dataset.
- Hybrid CTC/Attention end-to-end ASR:
    - Many available encoders: CRDNN (VGG + {LSTM,GRU,LiGRU} + DNN), ResNet, SincNet, vanilla transformers, contextnet-based transformers or conformers. Thanks to the flexibility of SpeechBrain, any fully customized encoder could be connected to the CTC/attention decoder and trained in few hours of work. The decoder is fully customizable as well: LSTM, GRU, LiGRU, transformer, or your neural network!
    - Optimised and fast beam search on both CPUs or GPUs.
- Transducer end-to-end ASR with a custom Numba loss to accelerate the training. Any encoder or decoder can be plugged into the transducer ranging from VGG+RNN+DNN to conformers.
- Pre-trained ASR models for transcribing an audio file or extracting features for a downstream task.

### Feature extraction and augmentation

SpeechBrain provides efficient and GPU-friendly speech augmentation pipelines and acoustic feature extraction:
- On-the-fly and fully-differentiable acoustic feature extraction: filter banks can be learned. This simplifies the training pipeline (you don't have to dump features on disk).
- On-the-fly feature normalization (global, sentence, batch, or speaker level).
- On-the-fly environmental corruptions based on noise, reverberation, and babble for robust model training.
- On-the-fly frequency and time domain SpecAugment.

### Speaker recognition, identification and diarization
SpeechBrain provides different models for speaker recognition, identification, and diarization on different datasets:
- State-of-the-art performance on speaker recognition and diarization based on ECAPA-TDNN models.
- Original Xvectors implementation (inspired by Kaldi) with PLDA.
- Spectral clustering for speaker diarization (combined with speakers embeddings).
- Libraries to extract speaker embeddings with a pre-trained model on your data.

### Speech enhancement and separation
- Recipes for spectral masking, spectral mapping, and time-domain speech enhancement.
- Multiple sophisticated enhancement losses, including differentiable STOI loss, MetricGAN, and mimic loss.
- State-of-the-art performance on speech separation with Conv-TasNet, DualPath RNN, and SepFormer.

### Multi-microphone processing
Combining multiple microphones is a powerful approach to achieve robustness in adverse acoustic environments:
- Delay-and-sum, MVDR, and GeV beamforming.
- Speaker localization.

### Performance
The recipes released with speechbrain implement speech processing systems with competitive or state-of-the-art performance. In the following, we report the best performance achieved on some popular benchmarks:

| Dataset        | Task           | System  | Performance  |
| ------------- |:-------------:| -----:|-----:|
| LibriSpeech      | Speech Recognition | CNN + Transformer | WER=2.46% (test-clean) |
| TIMIT      | Speech Recognition | CRDNN + distillation | PER=13.1% (test) |
| TIMIT      | Speech Recognition | wav2vec2 + CTC/Att. | PER=8.04% (test) |
| CommonVoice (French) | Speech Recognition | CRDNN | WER=17.7% (test) |
| VoxCeleb2      | Speaker Verification | ECAPA-TDNN | EER=0.69% (vox1-test) |
| AMI      | Speaker Diarization | ECAPA-TDNN | DER=2.13% (lapel-mix)|
| VoiceBank      | Speech Enhancement | MetricGAN+| PESQ=3.08 (test)|
| WSJ2MIX      | Speech Separation | SepFormer| SDRi=22.6 dB (test)|
| WSJ3MIX      | Speech Separation | SepFormer| SDRi=20.0 dB (test)|

For more details, take a look into the corresponding implementation in recipes/dataset/.

### Documentation & Tutorials
SpeechBrain is designed to speed-up research and development of speech technologies. Hence, our code is backed-up with three different levels of documentation:
- **Low-level:** during the review process of the different pull requests, we are focusing on the level of comments that are given. Hence, any complex functionality or long pipeline is supported with helpful comments enabling users to handily customize the code.
- **Functional-level:** all classes in SpeechBrain contains a detailed docstring that details the input and output formats, the different arguments, the usage of the function, the potentially associated bibliography, and a function example that is used for test integration during pull requests. Such examples can also be used to manipulate a class or a function to properly understand what is exactly happening.
- **Educational-level:** we provide various Google Colab (i.e. interactive) tutorials describing all the building-blocks of SpeechBrain ranging from the core of the toolkit to a specific model designed for a particular task. The number of available tutorials is expected to increase over time.

### Under development
We are currently working towards integrating DNN-HMM for speech recognition and machine translation.

# Quick installation

SpeechBrain is constantly evolving. New features, tutorials, and documentation will appear over time.
SpeechBrain can be installed via PyPI to rapidly use the standard library. Moreover,  a local installation can be used by those users that what to run experiments and modify/customize the toolkit. SpeechBrain supports both CPU and GPU computations. For most all the recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.


## Install via PyPI

Once you have created your Python environment (Python 3.8+) you can simply type:

```
pip install speechbrain
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

## Install with GitHub

Once you have created your Python environment (Python 3.8+) you can simply type:

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

# Learning SpeechBrain

Instead of a long and boring README, we prefer to provide you with different resources that can be used to learn how to customize SpeechBrain to adapt it to your needs:
- General information can be found on the [website](https://speechbrain.github.io).
- We offer many tutorials, you can start out from the [basic ones](https://speechbrain.github.io/tutorial_basics.html) about SpeechBrain basic functionalities and building blocks. We provide also more advanced tutorials (e.g SpeechBrain advanced, signal processing ...). You can browse them via the Tutorials drop down menu on [SpeechBrain website](https://speechbrain.github.io) in the upper right.
- Details on the SpeechBrain API, how to contribute, and the code are given in the [documentation](https://speechbrain.readthedocs.io/en/latest/index.html).

# License
SpeechBrain is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Also note that this project has no connection to the Apache Foundation, other than that we use the same license terms.
