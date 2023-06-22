# CL-MASR: A Continual Learning Benchmark for Multilingual ASR

This recipe includes scripts to train [Whisper](https://cdn.openai.com/papers/whisper.pdf) and
[WavLM](https://arxiv.org/abs/2110.13900)-based ASR systems on a subset of Common Voice 13 in a continual learning
fashion using a handful of methods including rehearsal-based, architecture-based, and regularization-based approaches.

The goal is to continually learn new languages while limiting forgetting on the previously learnt ones.
An ideal method should achieve both positive forward transfer (i.e. improve performance on new tasks leveraging
shared knowledge from previous tasks) and positive backward transfer (i.e. improve performance on previous tasks
leveraging shared knowledge from new tasks).

The following algorithms have been implemented so far:
- [Experience Replay (ER)](https://arxiv.org/abs/1811.11682)
- [Averaged Gradient Episodic Memory (A-GEM)](https://arxiv.org/abs/1812.00420)
- [Progressive Neural Networks (PNN)](https://arxiv.org/abs/1606.04671)
- [Piggyback (PB)](https://arxiv.org/abs/1801.06519)
- [Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796)
- [Learning without Forgetting (LwF)](https://arxiv.org/abs/1606.09282)

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

Download the CL-MASR benchmark data from [here](https://zenodo.org/record/8065754) and extract them
to a data directory of your choice (`CL-MASR` by default).
Navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr/<model>`, open a terminal and run:

```bash
python train_<cl-method>.py hparams/train_<cl-method>.yaml --data_dir <path-to-data-directory>
```

**NOTE**: to profile the model (optional), install `ptflops` and `torchinfo` as additional dependencies.

### Analyzing the results

Navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr`, open a terminal and run:

```bash
python analyze_logs.py <path-to-directory-containing-model-logs>
```

This command will recursively retrieve and analyze all log files that are named according to the
format `<cl-method>_base=<comma-separated-base-locales>_new=<comma-separated-new-locales>.txt`
(this is the default naming convention followed in all the training scripts).
You can find the resulting performance metric summaries and plots in `<path-to-directory-containing-model-logs>`.
See the help (`python analyze_logs.py -h`) for advanced configuration options.

**NOTE**: make sure to specify the `--im_refs` and `--fwt_refs` arguments that correspond to the given model (default to Whisper large-v2).

**NOTE**: to plot the results (optional), install `matplotlib` and/or `plotly` as additional dependencies.

---------------------------------------------------------------------------------------------------------

## 📈️ Results

Raw experiment logs are available [here](https://www.dropbox.com/scl/fo/5iedub8pie66y9bkhrexr/h?dl=0&rlkey=i6jy0r1ryg6oks4xs35wh6o1d).
We do not include the checkpoints due to storage limits (each experiment with Whisper large-v2 generates ~125 GB of checkpoint data).

Analyses generated via `analyze_logs.py` are available [here](https://www.dropbox.com/scl/fo/hsmowtbjdw4z17o5dhbwr/h?dl=0&rlkey=9yogen3bw8y8xmlupn2lxnn14).

All the experiments were run on five CentOS Linux 7 machines with an Intel(R) Xeon(R) Silver 4216 Cascade Lake CPU
with 32 cores @ 2.10 GHz, 64 GB RAM and an NVIDIA Tesla V100 SXM2 @ 32 GB with CUDA Toolkit 11.4.
With the specified hardware configuration, approximately one week is necessary to complete all the experiments.

**NOTE**: the checkpoint for WavLM large pretrained on the base languages is available [here](https://www.dropbox.com/scl/fo/f827hte231ddji0cl5d31/h?dl=0&rlkey=w8r2y01kbbv11041t0x6vvorh).

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
