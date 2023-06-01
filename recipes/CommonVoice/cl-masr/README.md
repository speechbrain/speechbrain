# CL-MASR: A Continual Learning Benchmark for Multilingual ASR

This recipe includes scripts to train [Whisper](https://cdn.openai.com/papers/whisper.pdf) and
[WavLM](https://arxiv.org/abs/2110.13900)-based ASR systems on Common Voice 13 in a continual learning fashion
using a handful of methods including regularization-based, replay-based and parameter isolation approaches.
The goal is to continually learn new languages while limiting forgetting on the previously learnt ones.
An ideal method should achieve both positive forward transfer (i.e. improve performance on new tasks leveraging
shared knowledge from previous tasks) and positive backward transfer (i.e. improve performance on previous tasks
leveraging shared knowledge from new tasks).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Clone the repository, navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr`,
open a terminal and run:

```bash
pip install -e ../../../                 # Install local SpeechBrain package
pip install -r extra-requirements.txt    # Install additional dependencies
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

### Running an experiment

Manually download the required Common Voice 13 locales (`en`, `zh-CN`, `de`, `es`, `ru`, `fr`, `pt`, `ja`,
`tr`, `pl`, `rw`, `eo`, `kab`, `lg`, `mhr`, `ckb`, `ab`, `kmr`, `fy-NL`, `ia`) from the [official
website](https://commonvoice.mozilla.org/en/datasets) and extract them to a common directory.
Navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr/<model>`, open a terminal and run:

```bash
python train_<cl-method>.py hparams/train_<cl-method>.yaml --data_dir <path-to-data-directory>
```

**NOTE**: to profile the model (optional), install `ptflops` and `torchinfo` as additional dependencies.

### Analyzing the results

Collect all `train_log.txt` files from each experiment, rename them according to the format
`<method-name>_base=<comma-separated-base-locales>_new=<comma-separated-new-locales>` and copy them
to a common directory.
Navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr`, open a terminal and run:

```bash
python analyze_logs.py <path-to-logs-directory>
```

You can find the performance metrics summaries in `<path-to-logs-directory>`.
See the help (`python analyze_logs.py -h`) for advanced configuration options.

**NOTE**: to plot the results (optional), install `matplotlib` and/or `plotly` as additional dependencies.

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
