# Continual Learning for Massively Multilingual Speech Recognition

This recipe includes scripts to train [Whisper](https://cdn.openai.com/papers/whisper.pdf) and
[WavLM](https://arxiv.org/abs/2110.13900)-based ASR systems on Common Voice in a continual learning fashion
using a handful of methods including regularization-based, replay-based and parameter isolation approaches.
The goal is to continually learn new languages while limiting forgetting on the previously learnt ones.
An ideal method should achieve both positive forward transfer (i.e. improve performance on new tasks leveraging
shared knowledge from previous tasks) and positive backward transfer (i.e. improve performance on previous tasks
leveraging shared knowledge from new tasks).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Clone the repository, navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning`,
open a terminal and run:

```
pip install -e ../../../                 # Install local SpeechBrain package
pip install -r extra-requirements.txt    # Install additional dependencies
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Manually download the required Common Voice locales (`en`, `zh-CN`, `de`, `es`, `ru`, `fr`, `pt`, `ja`,
`tr`, `pl`, `rw`, `eo`, `kab`, `lg`, `mhr`, `ckb`, `ab`, `kmr`, `fy-NL`, `ia`) from the [official
website](https://commonvoice.mozilla.org/en/datasets) and extract them to a common directory.
Navigate to `<path-to-repository>/recipes/CommonVoice/continual-learning/<model>`, open a terminal and run:

```
python train_<cl-method>.py hparams/train_<cl-method>.yaml --data_dir <path-to-data-directory>
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
