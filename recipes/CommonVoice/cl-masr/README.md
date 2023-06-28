# CL-MASR: A Continual Learning Benchmark for Multilingual ASR

This recipe includes scripts to train [Whisper](https://cdn.openai.com/papers/whisper.pdf) and
[WavLM](https://arxiv.org/abs/2110.13900)-based ASR systems on a subset of 20 languages selected from [Common Voice 13](https://commonvoice.mozilla.org/en/datasets)
in a continual learning fashion using a handful of methods including rehearsal-based, architecture-based, and regularization-based approaches.

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

Download the CL-MASR benchmark data (extracted from [Common Voice 13](https://commonvoice.mozilla.org/en/datasets), see [reference paper](https://arxiv.org/abs/1912.06670)) from [here](https://zenodo.org/record/8065754) and extract them
to a data folder of your choice (`CL-MASR` by default).
Navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr/<model>`, open a terminal and run:

```bash
python train_<cl-method>.py hparams/train_<cl-method>.yaml --data_folder <path-to-data-folder>
```

**NOTE**: to profile the model (optional), install `ptflops` and `torchinfo` as additional dependencies.

### Analyzing the results

Navigate to `<path-to-repository>/recipes/CommonVoice/cl-masr`, open a terminal and run:

```bash
python analyze_logs.py <path-to-folder-containing-model-logs>
```

This command will recursively retrieve and analyze all log files that are named according to the
format `<cl-method>_base=<comma-separated-base-locales>_new=<comma-separated-new-locales>.txt`
(this is the default naming convention followed in all the training scripts).
You can find the resulting performance metric summaries and plots in `<path-to-folder-containing-model-logs>`.
See the help (`python analyze_logs.py -h`) for advanced configuration options.

**NOTE**: make sure to specify the `--im_refs` and `--fwt_refs` arguments that correspond to the given model (default to Whisper large-v2).

**NOTE**: to plot the results (optional), install `matplotlib` and/or `plotly` as additional dependencies.

---------------------------------------------------------------------------------------------------------

## 📈️ Results

| Release  |         Hyperparameters         | Average AWER | Average BWT | Average IM | Average FWT |                                       Logs                                        | GPUs |
|:--------:|:-------------------------------:|:------------:|:-----------:|:----------:|:-----------:|:---------------------------------------------------------------------------------:| :--------:|
| 07-06-23 |  whisper/hparams/train_ft.yaml  |    98.50     |   -84.58    |   -4.16    |    -0.83    | [Link](https://www.dropbox.com/sh/gjthcje9i2rztsk/AABWcxRpyVek5VVLy1UIU5JUa?dl=0) | 1xV100 32GB |
| 07-06-23 |  whisper/hparams/train_er.yaml  |    50.83     |   -13.20    |   -0.81    |    -4.17    | [Link](https://www.dropbox.com/sh/3ykkqss8trf4mh0/AADKIGt_IbBpYy6z1zGMv9t5a?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_agem.yaml |    81.08     |   -55.85    |    0.20    |    -5.19    | [Link](https://www.dropbox.com/sh/x3inrfmktk5eqeu/AAAltNoaaiexezOjYD3J2H0Qa?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_pnn.yaml  |    44.12     |    0.00     |    3.18    |    -8.16    | [Link](https://www.dropbox.com/sh/k8zeoxpbh9yjngi/AACKXnZEIInWzNfN6aZWCd5ra?dl=0) | 1xV100 32GB |
| 07-06-23 |  whisper/hparams/train_pb.yaml  |    43.95     |    0.00     |    3.51    |    -8.50    | [Link](https://www.dropbox.com/sh/load8e6dwwl31kc/AAAoROiJLCu6haFJqJcZ_uyya?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_ewc.yaml  |    98.04     |   -68.32    |    2.87    |    -7.85    | [Link](https://www.dropbox.com/sh/ve00u3jwru880x7/AAAl5tjVa3K1F_JelMC_uimpa?dl=0) | 1xV100 32GB |
| 07-06-23 | whisper/hparams/train_lwf.yaml  |    95.76     |   -77.50    |    0.00    |    -4.98    | [Link](https://www.dropbox.com/sh/9z3ejbc371c36rk/AABypJbr782kVVOrqA0neEzxa?dl=0) | 1xV100 32GB |
| 07-06-23 |   wavlm/hparams/train_ft.yaml   |    91.61     |   -54.67    |   -10.19   |    -0.21    | [Link](https://www.dropbox.com/sh/hluabvm3ph0j7ee/AAAZswrK0KjstZm1Q5bb29Xfa?dl=0) | 1xV100 32GB |
 | 07-06-23 |   wavlm/hparams/train_er.yaml   |    60.79     |    -8.96    |   -7.62    |    -2.77    | [Link](https://www.dropbox.com/sh/1den1zq0md5rfgv/AABrgM_1O85WwXSBvyMwxYkha?dl=0) | 1xV100 32GB |
 | 07-06-23 |  wavlm/hparams/train_agem.yaml  |    72.54     |    13.59    |   35.29    |   -45.69    | [Link](https://www.dropbox.com/sh/cn737pp6tpupsy2/AACf309ybWRFStrCbdytx16ja?dl=0) | 1xV100 32GB |
 | 07-06-23 |  wavlm/hparams/train_pnn.yaml   |    66.07     |    0.00     |   12.95    |   -23.34    | [Link](https://www.dropbox.com/sh/jz9a64xriifilmf/AADHQYrbuHNe1-rDkV28H298a?dl=0) | 1xV100 32GB |
 | 07-06-23 |   wavlm/hparams/train_pb.yaml   |    61.87     |    0.00     |    2.75    |   -13.15    | [Link](https://www.dropbox.com/sh/wfj83oh8u8xru8e/AAAqmMjMs1tK1X0I53Ldk7c5a?dl=0) | 1xV100 32GB |
 | 07-06-23 |  wavlm/hparams/train_ewc.yaml   |    86.98     |   -39.54    |   -4.26    |    -6.13    | [Link](https://www.dropbox.com/sh/poi5n6bmw3g9xs3/AABNozqigh54fKUdyWbf_WLOa?dl=0) | 1xV100 32GB |
 | 07-06-23 |  wavlm/hparams/train_lwf.yaml   |    87.17     |   -26.03    |   10.42    |   -20.82    | [Link](https://www.dropbox.com/sh/fggjafxdrtux68y/AADpdUV1Ny2may-G3pNwUrB6a?dl=0) | 1xV100 32GB |

Raw experiment logs are available [here](https://www.dropbox.com/sh/y15vy2op74a5tbu/AACgtxN_uYRGfvCTtiUB7d_ma?dl=0).
We do not include the checkpoints due to storage limits (each experiment with Whisper large-v2 generates ~125 GB of checkpoint data).

Analyses generated via `analyze_logs.py` are available [here](https://www.dropbox.com/sh/0ndrp570vlsh893/AAC2WSZQu00ZducN80Ff5dWla?dl=0).

All the experiments were run on five CentOS Linux 7 machines with an Intel(R) Xeon(R) Silver 4216 Cascade Lake CPU
with 32 cores @ 2.10 GHz, 64 GB RAM and an NVIDIA Tesla V100 SXM2 @ 32 GB with CUDA Toolkit 11.4.
With the specified hardware configuration, approximately one week is necessary to complete all the experiments.

**NOTE**: the checkpoint for WavLM large pretrained on the base languages is available [here](https://www.dropbox.com/sh/3h4k8ccn465bv48/AABM7fCNOU9tTQPD0vCT8-K4a?dl=0).

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
