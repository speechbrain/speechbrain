# Continual Learning for Massively Multilingual Speech Recognition

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

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

To deactivate it, run:

```
conda deactivate
```

To permanently delete it, run:

```
conda env remove -n cl-env
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning/whisper`, open a terminal and run:

```
source activate cl-env
python train.py hparams/train_ar.yaml
```

**NOTE**: you can download Common Voice 11.0 beforehand, (requires ~600 GB of free space) and store them for later use.
To do so, navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning`, open a terminal and run:

```
source activate cl-env
python common_voice_prepare.py -l <list-of-locales-to-download> -d data
```

It is recommended to compress the downloaded datasets into `tar.gz` archives to store them more efficiently:

```
tar -czvf common_voice_11.tar.gz data
rm -r data
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
