# Profiling & benchmark as part of testing

_Recognition performance aside, are we real-time?_

Among the tools out there, PyTorch offers its profiler to benchmark computational time and memory demands.
SpeechBrain wraps this profiler into `@profile`, `@profile_optimiser`, and `@profile_analyst` decorators.
While our [tutorial](colab-url) suggests how to use them, this recipe helps to figure out a guiding estimate (take it on nominal level) for inference with pretrained models regarding:
* real-time factor
* peak memory

---

***In-scope:*** for researchers and developers to figure out if their systems go somewhat in the right direction.

***Out-scope:*** technology readiness level reporting.

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
* `speechbrain/asr-wav2vec2-commonvoice-fr` _(~3 minutes on a V100; 2.6s for each 32s x 8 batch)_<br/>is faster than<br/>`asr-crdnn-rnnlm-librispeech` _(~15 minutes on a V100; 13.1s for each 32s x 8 batch)_<br/>which checks 80 hypotheses.
* On CPU-only (dual-cores; 4 GB/CPU), `speechbrain/asr-wav2vec2-commonvoice-fr` benchmarks took ~4m _on 16x CPUs_; ~5m _on 8x CPUs_, and ~10m _on 4x CPU_.
* On 2x CPU (8 GB) and 1x CPU (4 GB), the full benchmark ran out-of-memory, so we skipped the _{5s,8s,32s} x 8_ and _{8s,32s} x 4_ configurations—profiling took, respectively, ~4m and ~7m.

# Results (inference only)

As an example (only); we report on the [speechbrain/asr-wav2vec2-commonvoice-fr](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-fr) pretrained model.<br/>_(We used a long LibriSpeech file to get a first picture of the situation.)_

***Note:*** out of six batches, only the last two are used for profiling.

## Real-time factors

As CPU + CUDA, in exponent-notation (real-time: order of magnitude < 0).

| 1x V100 | 1                   | 4                   | 8                   |
|--------:|:--------------------|:--------------------|:--------------------|
|      1s | 4.08E-02 + 7.18E-02 | 1.01E-02 + 2.72E-02 | 5.36E-03 + 1.72E-02 |
|      2s | 2.07E-02 + 2.04E-02 | 5.22E-03 + 1.04E-02 | 4.30E-03 + 8.72E-03 |
|      5s | 7.92E-03 + 1.29E-02 | 4.31E-03 + 8.82E-03 | 4.10E-03 + 8.28E-03 |
|      8s | 5.31E-03 + 1.12E-02 | 4.20E-03 + 8.54E-03 | 4.04E-03 + 8.17E-03 |
|     32s | 5.44E-03 + 1.12E-02 | 5.10E-03 + 1.05E-02 | 4.98E-03 + 1.02E-02 |

***Note:*** copying batches to the GPU also takes time.

---

| 16x CPU |        1 |        4 |         8 |
|--------:|---------:|---------:|----------:|
|      1s | 1.07E-01 | 5.99E-02 |  4.67E-02 |
|      2s | 6.22E-02 | 4.67E-02 |  3.73E-02 |
|      5s | 5.57E-02 | 3.69E-02 |  3.55E-02 |
|      8s | 5.01E-02 | 3.61E-02 |  3.77E-02 |
|     32s | 4.41E-02 | 4.33E-02 |  4.69E-02 |

| 4x CPU |          1 |        4 |         8 |
|-------:|-----------:|---------:|----------:|
|     1s |   3.24E-01 | 1.39E-01 |  1.15E-01 |
|     2s |   1.70E-01 | 1.20E-01 |  1.05E-01 |
|     5s |   1.36E-01 | 1.07E-01 |  1.06E-01 |
|     8s |   1.28E-01 | 1.10E-01 |  1.10E-01 |
|    32s |   1.38E-01 | 1.37E-01 |  1.39E-01 |

| 1x CPU |          1 |             4 |           8 |
|-------:|-----------:|--------------:|------------:|
|     1s |   7.83E-01 |      4.77E-01 |   4.17E-01  |
|     2s |   5.80E-01 |      4.22E-01 |    3.94E-01 |
|     5s |   4.76E-01 |      4.10E-01 | _(skipped)_ |
|     8s |   4.66E-01 |   _(skipped)_ | _(skipped)_ |
|    32s |   5.32E-01 |   _(skipped)_ | _(skipped)_ |


## Memory peaks

|       1x V100 | 1         | 4         | 8       |
|--------------:|:----------|:----------|:--------|
|            1s | 0.13 Gb   | 0.15 Gb   | 0.29 Gb |
|            2s | 0.13 Gb   | 0.29 Gb   | 0.59 Gb |
|            5s | 0.18 Gb   | 0.73 Gb   | 1.47 Gb |
|            8s | 0.29 Gb   | 1.17 Gb   | 2.35 Gb |
|           32s | 1.15 Gb   | 4.60 Gb   | 9.20 Gb |

| 16x CPU | 1       | 4       | 8       |
|--------:|:--------|:--------|:--------|
|      1s | 0.09 Gb | 0.18 Gb | 0.29 Gb |
|      2s | 0.10 Gb | 0.32 Gb | 0.62 Gb |
|      5s | 0.21 Gb | 0.76 Gb | 1.50 Gb |
|      8s | 0.32 Gb | 1.20 Gb | 2.38 Gb |
|     32s | 1.18 Gb | 4.63 Gb | 9.20 Gb |

***Note:*** these numbers discern overheads from handling the model itself, the data pipeline, python environments, etc. (the actual VRAM demand will be higher). This overview is purely about the inference step. The `profile.py` script adds its own overheads on top.

# Some pointers

Starting with the PyTorch profiler and benchmark visualisation:
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/
- https://github.com/pytorch/kineto/tree/main/tb_plugin

Where to go from here:
- https://horace.io/brrr_intro.html (thanks for the xref: @RuABraun !)
- https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html


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