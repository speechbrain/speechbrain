# The SpeechBrain Toolkit

<p align="center">
  <img src="http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/img/logo_noname_rounded_big.png" alt="drawing" width="250"/>
</p>

SpeechBrain is an **open-source** and **all-in-one** speech toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **multi-microphone signal processing** and many others.

*SpeechBrain is currently in beta*.

| **[Tutorials](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/tutorial_basics.html)** | **[Website](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/)** | **[Documentation](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/documentation/index.html)** | **[Contributing](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/documentation/contributing.html)** |

# Key features

### Feature extraction and augmentation
SpeechBrain provides efficient and GPU-friendly speech augmentation pipelines and acoustic feature extraction:
- On-the-fly and fully-differentiable acoustic features extraction: filter banks can be learnt. This also simplify the training pipeline as speech augmentation can be performed on-the-fly. This facilitate the integration of fully end-to-end encoders directly dealing with the raw waveform.
- On-the-fly features normalisation (global, sentence, bath or speaker level).
- On-the-fly environmental corruptions based on noise, reverberation and babble for robust model training.
- On-the-fly frequency domain and time domain SpecAugment.

### Speech recognition

SpeechBrain supports state-of-the-art methods for end-to-end speech recognition:
- State-of-the-art performance or at least comparable with other existing toolkits in several ASR benchmarks.
- Easily customisable neural language models including RNNLM and TransformerLM. We also propose few pretrained models to save you computations (more to come!). We compose with hugging face `dataset` to facilitate the training over large text dataset.
- Hybrid CTC/Attention end-to-end ASR:
    - Many available encoders: CRDNN (VGG + {LSTM,GRU,LiGRU} + DNN), ResNet, SincNet, vanilla transformers, contextnet-based transformers or conformers. Thanks to the flexibility of SpeechBrain, any fully customised mono or multi-encoder could be connected to the CTC/attention decoder and trained in few hours of work. Note that the decoder is also fully customisable: LSTM, GRU, LiGRU, transformer or your handcrafted neural network!
    - Optimised beam search and greedy decoding significantly faster than existing toolkit at decoding time. Our decoding can be performed both with CPU or GPU.
- Transducer end-to-end ASR with a custom Numba loss to accelerate the training. Any encoder or decoder can be plugged into the transducer ranging from VGG+RNN+DNN to conformers.
- Scheme to simply use a trained ASR model to transcribe an audio file.

### Speaker recognition
SpeechBrain provides different models for speaker recognition, identification and diarization on different datasets:
- State-of-the-art performance on speaker recognition and diarization based on ECAPA-TDNN models.
- Original Xvectors implementation (inspired by Kaldi) with PLDA.
- Contrastive learning based training for speaker recognition.
- Spectral clustering for speaker diarization (combined with speakers embeddings).
- Scheme to simply use a trained speaker embeddings extractor to obtain embeddings from audio files.

### Speech enhancement
Spectral masking, spectral mapping, and time-domain enhancement are different methods already available within SpeechBrain. Separation methods such as Conv-TasNet, DualPath RNN, and SepFormer are implemented as well.

### Multi-microphone processing
Combining multiple microphones is a powerful approach to achieve robustness in adverse acoustic environments. SpeechBrain provides various techniques for beamforming (e.g, delay-and-sum, MVDR, and GeV) and speaker localization.

### Research & development
SpeechBrain is designed to speed-up research and development of speech technologies. It is modular, flexible, easy-to-customize, and contains several recipes for popular datasets. Documentation and tutorials are here to help newcomers using SpeechBrain.

### Under development
We currently are working towards integrating DNN-HMM for speech recognition, speech translation, and machine translation.

# Quick installation

SpeechBrain is constantly evolving. New features, tutorials, and documentation will appear over time.
SpeechBrain can be installed via PyPI to rapidly use the standard library. Moreover,  a local installation can be used by those users that what to run experiments and modify/customize the toolkit. SpeechBrain supports both CPU and GPU computations. For most all the recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.


## Install via PyPI

Once you have created your python environment (Python 3.8+) you can simply type:

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple BeechSprain
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

## Install with GitHub

Once you have created your python environment (Python 3.8+) you can simply type:

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

Instead of a long and boring README, we prefer to provide you with different resources that can be used to learn how to customise SpeechBrain to adapt it to your needs:

- General information and Google Colab tutorials can be found on the [website](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/).
- Many [tutorials](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/tutorial_basics.html) about the basics of SpeechBrain.
- Details on the SpeechBrain API, how to contribute and the code are given the [documentation](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/documentation/index.html).

# License
SpeechBrain is released under the Apache license, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Also note that this project has no connection to the Apache Foundation, other than that we use the same license terms.
