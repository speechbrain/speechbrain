# Profiling & benchmark as part of testing

_Recognition performance aside, are we real-time?_

Among the tools out there, PyTorch offers its profiler to benchmark computational time and memory demands.
SpeechBrain wraps this profiler into `@profile`, `@profile_optimiser`, `@profile_analyst`, and `@profile_report` decorators.
While our [tutorial](https://colab.research.google.com/drive/1X9eeAEy19BgEJX4YZWjo1Huku_8cOUGJ?usp=sharing) suggests how to use them, this recipe helps to figure out a guiding estimate (take it on nominal level) for inference with pretrained models regarding:
* real-time factor
* peak memory

This tool uses the `@profile_report` decorator which anticipates real-time profiling as:
1. simulate ten batches per duration/batch size setting, ignore first three recordings
2. compute average and standard deviation of the remaining seven profiled recordings
3. report real-time by its upper control limit (µ + 3σ); instead of its expectation only

About 97.5% of attempts to reproduce (with similar hardware, set-up, etc.) reported figures should remain consistent.

---

***In-scope:*** for researchers and developers to figure out if their systems go somewhat in the right direction.

***Out-scope:*** technology readiness level reporting.

***Note:*** requires PyTorch >= 1.10; the profiler of earlier versions became a legacy profiler.

# How to run

`python profile.py profile.yaml`

Specify benchmark configurations in the [profile.yaml file](profile.yaml)
```YAML
# Which model to profile?
pretrained_model:
  source: speechbrain/asr-wav2vec2-commonvoice-fr  # HuggingFace or local path
  type: EncoderASR  # Pretrained interface

# Which settings should be benchmarked?
profiling_dimensions:
  audio_mockup_secs: [1, 2, 5, 8, 32]  # 1s, 2s, ...
  batch_sizes: [1, 4, 8]  # 1 file per batch, 4 files per batch, ...
  # Fancy to truncate/repeat a real audio?
  example_audio: ../../samples/audio_samples/example2.flac # None -> random data
  # Some audio_mockup_secs x batch_sizes configs can get VRAM intensive
  triangle_only: False  # True -> skip the heavy ones, e.g., 32s x 8 files
  export_logs: True  # export trace logs for visualisation
```

***Note:*** The choice of a real audio over mock-up/random data is to clarify on the impact of recursive computations whose paths depend on data, e.g., beamforming. _(RNN language models are fast on noise: useful for quick inquiry on memory peaks only; not for inquiry on real-time factors.)_


# Execution time of profiling

How long one test takes depends on its configuration, the pretrained model, and on the machine.

On a 1x V100 GPU, profiling `speechbrain/asr-wav2vec2-commonvoice-fr` run for ~4 minutes.
By contrast, profiling `speechbrain/asr-crdnn-rnnlm-librispeech` (tracks 80 hypotheses) run for ~24 minutes.
(One might consider to track way less hypotheses.)



For low-memory machines, it might be good to benchmark only a few data points of the
full duration vs batch size table (`triangle_only=True`).
More demanding settings might not be satisfiable by VRAM.

# Results (inference only)

As an example (only); we report on the [speechbrain/asr-wav2vec2-commonvoice-fr](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr) pretrained model.
<br/>_(We used a long LibriSpeech file to get a first picture of the situation.)_


## Real-time factors (upper control limits)

As CPU + CUDA, in E-[xponent] notation (real-time: number after the `E` is negative).

_Note: 1E0 = 1x 10^0 = 1.0 is real-time equivalent, every factor below has a negative exponent._

> `5.15E-02 + 3.62E-02` means
> * CPU factor: 5.15E-02 is real-time (factor: 0.0515 < 1)
> * CUDA factor: 3.62E-02 is real-time (factor: 0.0362 < 1)
> * Total: 1.26E-01 (0.126 < 1) -> real-time

| 1x V100 |    batch size: 1    |         4           |         8           |
|:-------:|:-------------------:|:-------------------:|:-------------------:|
|   1s    | 5.15E-02 + 3.62E-02 | 1.69E-02 + 1.38E-02 | 7.28E-03 + 9.62E-03 |
|   2s    | 2.77E-02 + 1.04E-02 | 6.54E-03 + 5.48E-03 | 4.53E-03 + 4.56E-03 |
|   5s    | 1.28E-02 + 6.93E-03 | 4.54E-03 + 4.62E-03 | 4.31E-03 + 4.34E-03 |
|   8s    | 8.17E-03 + 5.90E-03 | 4.45E-03 + 4.52E-03 | 4.25E-03 + 4.32E-03 |
|   32s   | 5.66E-03 + 5.81E-03 | 5.29E-03 + 5.46E-03 | 5.15E-03 + 5.27E-03 |

***Note:*** `cudaMemcpyAsync` is the time-demanding function event on CPU.

---

On dual-core CPUs.

| 16x CPU |        1 |        4 |        8 |
|:-------:|:--------:|:--------:|:--------:|
|   1s    | 1.15E-01 | 6.40E-02 | 4.42E-02 |
|   2s    | 7.57E-02 | 4.43E-02 | 2.76E-02 |
|   5s    | 5.22E-02 | 2.64E-02 | 2.18E-02 |
|   8s    | 4.77E-02 | 2.42E-02 | 3.03E-02 |
|   32s   | 3.77E-02 | 3.34E-02 | 2.74E-02 |

| 8x CPU |        1 |        4 |        8 |
|:------:|:--------:|:--------:|:--------:|
|   1s   | 1.28E-01 | 7.88E-02 | 4.37E-02 |
|   2s   | 7.47E-02 | 4.78E-02 | 2.74E-02 |
|   5s   | 5.27E-02 | 2.67E-02 | 2.17E-02 |
|   8s   | 4.15E-02 | 2.61E-02 | 2.63E-02 |
|  32s   | 3.69E-02 | 2.83E-02 | 2.62E-02 |

| 4x CPU |        1 |        4 |        8 |
|:------:|:--------:|:--------:|:--------:|
|   1s   | 1.26E-01 | 6.56E-02 | 4.63E-02 |
|   2s   | 6.55E-02 | 4.75E-02 | 3.32E-02 |
|   5s   | 5.72E-02 | 3.91E-02 | 2.98E-02 |
|   8s   | 5.11E-02 | 3.26E-02 | 2.57E-02 |
|  32s   | 3.88E-02 | 3.00E-02 | 3.08E-02 |

| 2x CPU |        1 |        4 |        8 |
|:------:|:--------:|:--------:|:--------:|
|   1s   | 1.24E-01 | 6.62E-02 | 4.36E-02 |
|   2s   | 7.57E-02 | 4.65E-02 | 2.64E-02 |
|   5s   | 4.97E-02 | 3.60E-02 | 3.04E-02 |
|   8s   | 3.94E-02 | 3.12E-02 | 2.72E-02 |
|  32s   | 2.89E-02 | 3.34E-02 | 3.15E-02 |

| 1x CPU |        1 |    4     |    8     |
|:------:|:--------:|:--------:|:--------:|
|   1s   | 1.21E-01 | 7.67E-02 | 4.54E-02 |
|   2s   | 7.32E-02 | 4.46E-02 | 2.70E-02 |
|   5s   | 5.14E-02 | 2.64E-02 |  _skip_  |
|   8s   | 4.05E-02 |  _skip_  |  _skip_  |
|  32s   | 2.97E-02 |  _skip_  |  _skip_  |

_Note: these values report upper control limits, averages under the impact of deviation. The 16x CPU benchmark appears inconsistent for this setting of durations and batch sizes; a reason could be that more extensive hardware might be suited better for even heavier computations (lower workloads are not processed at full efficiency)._


## Memory peaks

| 1x V100 |              1 |              4 |              8 |
|:-------:|:--------------:|:--------------:|:--------------:|
|   1s    | 0.00 + 0.13 Gb | 0.00 + 0.15 Gb | 0.00 + 0.29 Gb |
|   2s    | 0.00 + 0.13 Gb | 0.00 + 0.29 Gb | 0.00 + 0.59 Gb |
|   5s    | 0.00 + 0.18 Gb | 0.00 + 0.73 Gb | 0.00 + 1.47 Gb |
|   8s    | 0.00 + 0.29 Gb | 0.00 + 1.17 Gb | 0.00 + 2.35 Gb |
|   32s   | 0.00 + 1.15 Gb | 0.00 + 4.60 Gb | 0.00 + 9.20 Gb |

| 16x CPU |       1 |       4 |       8 |
|:-------:|:-------:|:-------:|:-------:|
|   1s    | 0.09 Gb | 0.18 Gb | 0.32 Gb |
|   2s    | 0.10 Gb | 0.32 Gb | 0.62 Gb |
|   5s    | 0.21 Gb | 0.76 Gb | 1.50 Gb |
|   8s    | 0.32 Gb | 1.20 Gb | 2.38 Gb |
|   32s   | 1.15 Gb | 4.63 Gb | 9.23 Gb |

***Note:*** these numbers discern overheads from handling the model itself, the data pipeline, python environments, etc. (the actual VRAM demand will be higher). This overview is purely about the inference step. The `profile.py` script adds its own overheads on top.


## Contrastive results

For comparison on 1x V100: [speechbrain/asr-crdnn-rnnlm-librispeech](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech).

|  Real-time factor   |                   1 |                   4 |                   8 |
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|         1s          | 2.32E-01 + 8.76E-02 | 7.71E-02 + 3.51E-02 | 6.17E-02 + 3.07E-02 |
|         2s          | 1.36E-01 + 4.35E-02 | 3.92E-02 + 1.86E-02 | 3.33E-02 + 1.62E-02 |
|         5s          | 1.32E-01 + 6.01E-02 | 4.24E-02 + 2.84E-02 | 3.65E-02 + 2.52E-02 |
|         8s          | 1.03E-01 + 5.21E-02 | 4.18E-02 + 2.64E-02 | 3.28E-02 + 2.70E-02 |
|         32s         | 6.81E-02 + 3.99E-02 | 3.22E-02 + 2.69E-02 | 2.77E-02 + 2.69E-02 |

| Memory peaks |              1 |              4 |              8 |
|:------------:|:--------------:|:--------------:|:--------------:|
|      1s      | 0.00 + 0.78 Gb | 0.00 + 0.78 Gb | 0.00 + 0.78 Gb |
|      2s      | 0.00 + 0.78 Gb | 0.00 + 0.78 Gb | 0.00 + 0.78 Gb |
|      5s      | 0.00 + 0.78 Gb | 0.00 + 0.78 Gb | 0.00 + 0.92 Gb |
|      8s      | 0.00 + 0.78 Gb | 0.00 + 0.78 Gb | 0.00 + 1.46 Gb |
|     32s      | 0.00 + 0.78 Gb | 0.00 + 2.87 Gb | 0.00 + 5.74 Gb |

_Note: here, the RNN LM calls up to: 24,731x `aten::copy_` (882x for the model above) and 22,4940x `cudaLaunchKernel`. (This could be because of tensors being created on CPU and then moved to cuda with `.to()` instead of creating them on the device right away.)_


# Some pointers

Starting with the PyTorch profiler and benchmark visualisation:
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/
- https://github.com/pytorch/kineto/tree/main/tb_plugin

Where to go from here:
- https://horace.io/brrr_intro.html (thanks for the xref: @RuABraun !)
- https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html
- https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/profile.html

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
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
