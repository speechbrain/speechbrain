# Continual Learning for Massively Multilingual Speech Recognition

This recipe includes scripts to train an [OpenAI Whisper](https://cdn.openai.com/papers/whisper.pdf)-based ASR system
on Common Voice (version 12.0 by default) in a continual learning fashion using a handful of methods including
regularization-based, replay-based, parameter isolation and prompt-based approaches.
The goal is to continually learn new languages while limiting forgetting on the previously learnt ones.
An ideal method should achieve both positive forward transfer (i.e. improve performance on new tasks leveraging shared
knowledge from previous tasks) and positive backward transfer (i.e. improve performance on previous tasks leveraging
shared knowledge from new tasks).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

### Using Conda (recommended)

First of all, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Clone the repository, navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning`, open a terminal and run:

```
conda env create -f environment.yaml
```

Project dependencies (pinned to a specific version to reduce compatibility and reproducibility issues)
will be installed in a [Conda](https://www.anaconda.com/) virtual environment named `cl-env`.
To activate it, run:

```
source activate cl-env
```

or

```
conda activate cl-env
```

To deactivate it, run:

```
conda deactivate
```

To permanently delete it, run:

```
conda env remove -n cl-env
```

### Using Pip

Clone the repository, navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning`, open a terminal and run:

```
pip install -e ../../../             # Install local SpeechBrain package
pip install -r extra-requirements    # Install additional dependencies
```

Note that with this approach compatibility and reproducibility are not guaranteed.

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning/whisper`, open a terminal and run
(remember to activate the virtual environment via `source activate cl-env` if you installed the project using Conda):

```
python train_<cl-method>.py hparams/train_<cl-method>.yaml
```

**NOTE**: you can download Common Voice 12.0 beforehand, (requires ~600 GB of free space) and store them for later use.
To do so, navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning`, open a terminal and run:

```
python common_voice_prepare.py -l <list-of-locales-to-download> -d data
```

It is recommended to compress the downloaded datasets into `tar.gz` archives to store them more efficiently:

```
tar -czvf common_voice_12.tar.gz data
rm -r data
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
