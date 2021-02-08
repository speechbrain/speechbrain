# The SpeechBrain Toolkit

<img src="https://speechbrain.github.io/assets/logo_noname_rounded_small.png" alt="drawing" width="200"/>

SpeechBrain is an **open-source** and **all-in-one** speech toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **multi-microphone signal processing** and many others.

*SpeechBrain is currently in beta*.

| **[Tutorials](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/tutorial_basics.html)** | **[Website](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/)** | **[Documentation](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/documentation/index.html)** | **[Contributing](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/documentation/contributing.html)** |

# Key features

## Speech Recognition
SpeechBrain supports state-of-the-art methods for end-to-end speech recognition, including models based on CTC, CTC+Attention, transducers, Transformers, and neural language models relying on recurrent neural networks and transformers.

## Speaker Recognition
Speaker recognition is already deployed in a wide variety of realistic applications. SpeechBrain provides different models for speaker recognition, including X-vector, ECAPA-TDNN, PLDA, contrastive learning.

## Speech Enhancement
Spectral masking, spectral mapping, and time-domain enhancement are different methods already available within SpeechBrain. Separation methods such as ConvTasnet, DualPath RNN, and SepFormer are implemented as well.

## Speech Processing
Speechbrain provides efficient and GPU-friendly speech augmentation pipelines and acoustic features extraction, normalisation that can be used on-the-fly during your experiment.

## Multi Microphone Processing
Combining multiple microphones is a powerful approach to achieve robustness in adverse acoustic environments. SpeechBrain provides various techniques for beamforming (e.g, delay-and-sum, MVDR, and GeV) and speaker localization.

## Research & Development

SpeechBrain is designed to speed-up research and development of speech technologies. It is modular, flexible, easy-to-customize, and contains several recipes for popular datasets. Documentation and tutorials are here to help newcomers using SpeechBrain.

## Under development

We currently are working towards integrating DNN-HMM for speech recognition, end-to-end spoken language understanding, speech translation and machine translation.

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

Instead of a long and boring ReadMe, we prefer to provide you with different resources that can be used to learn how to customise SpeechBrain to adapt it to your needs:

- General information and Google Colab tutorials can be found on the [website](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/).
- Many [tutorials](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/tutorial_basics.html) about the basics of SpeechBrain.
- Details on the SpeechBrain API, how to contribute and the code are given the [documentation](http://www.darnault-parcollet.fr/Parcollet/hiddennoshare/speechbrain.github.io/documentation/index.html).

# License
SpeechBrain is released under the Apache license, version 2.0. The Apache license is a popular BSD-like license. SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances you may have to distribute a license document). Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Also note that this project has no connection to the Apache Foundation, other than that we use the same license terms.
