# GigaSpeech streaming and non streaming speech recognition with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the GigaSpeech dataset.
Before running this recipe, make sure numba is installed (pip install numba)

## Data access and download

**The XL set is fairly large, 2.2TB are necessary to store the compressed and uncompressed version of the data**

SpeechBrain supports two ways of dealing with the GigaSpeech dataset:
1. [HuggingFace dataset](https://huggingface.co/datasets/speechcolab/gigaspeech/). For HuggingFacem note that **you must use** the HuggingFace client to log in first before running the recipe.
2. [Original Github](https://github.com/SpeechColab/GigaSpeech).

You simply need to follow the instructions on either of the above links. **We strongly
recomment using HuggingFace as the download speed for people outside of China is
much quicker**.

## Data preparation

**This step can be very long depending on your internet connection and filesystem for the XL split of GigaSpeech. For DDP (multi GPU) the recipe must be run once without DDP otherwise it will timeout. You do not want to let X GPUs hang out without doing nothing for hours anyway. Use the *data_prep_only* flag from the yaml to exit after data preparation**

SpeechBrain will automatically download the dataset if you use HuggingFace. Note that if you use HuggingFace, the *data_folder* argument is used to store the **extracted** dataset. However, HuggingFace first needs to download the compressed data, and this is not stored in *data_folder* by default. Indeed, HuggingFace is a bit strict in the way it operates with dataset, and the data will be put into the folder specified by the environment variable *HF_HUB_CACHE* or, if not set, *HF_HOME* or, if not set, *XDG_CACHE_HOME*. Hence, we recommend setting the *HF_HUB_CACHE* to the place where you want to store the data first. For example, you can set it like this:

```export HF_HUB_CACHE=/path/to/your/data/folder```

# Extra-Dependencies
This recipe supports two implementations of the transducer loss, see `use_torchaudio` arg in the yaml file:
1. Transducer loss from torchaudio (this requires torchaudio version >= 0.10.0).
2. Speechbrain implementation using Numba. To use it, please set `use_torchaudio=False` in the yaml file. This version is implemented within SpeechBrain and  allows you to directly access the python code of the transducer loss (and directly modify it if needed).

The Numba implementation is currently enabled by default as the `use_torchaudio` option is incompatible with `bfloat16` training.

Note: Before running this recipe, make sure numba is installed. Otherwise, run:
```
pip install numba
```

# How to run it
```shell
python train.py hparams/conformer_transducer.yaml
```

## Precision Notes
If your GPU effectively supports fp16 (half-precision) computations, it is recommended to execute the training script with the `--precision=fp16` (or `--precision=bf16`) option.
Enabling half precision can significantly reduce the peak VRAM requirements. For example, in the case of the Conformer Transducer recipe trained with GigaSpeech, the peak VRAM decreases from 39GB to 12GB when using fp16.
According to our tests, the performance is not affected.

## Streaming model

# Results (non-streaming)

Results are obtained with beam search and no LM (no-streaming i.e. full context).


| Release       |    LM | Val. CER | Val. WER | Test CER | Test WER | Model | GPUs |
|:-------------:| -----:| --------:| --------:| --------:| --------:| :---------:|:-----------:|
| 08-11-2024    |  None | 6.09%\*  | 11.75%\* | 6.14%\*  | 11.97%\* | [Dropbox](https://www.dropbox.com/scl/fo/jg0vzm8l27o9qsixpqzjo/ABpKqmTMg24RVJKLY5Io1eU?rlkey=8z51y0gosme1fh4niahvi6b84&st=6smf7i5z&dl=0), [HuggingFace](https://huggingface.co/speechbrain/asr-streaming-conformer-gigaspeech) | 4x A100 80GB |

\*: These results were obtained with our usual training scripts and are included for completeness, **but note that we have noticed an unexpected significant improvement to the error rate (see #2753) using the inference code path. Please refer to the table below for better and more accurate results.**

### WER vs chunk size & left context

The following matrix presents the Word Error Rate (WER%) achieved on GigaSpeech
`test` with various chunk sizes (in ms).

The relative difference is not trivial to interpret, because we are not testing
against a continuous stream of speech, but rather against utterances of various
lengths. This tends to bias results in favor of larger chunk sizes.

The chunk size might not accurately represent expected latency due to slight
padding differences in streaming contexts.

The left chunk size is not representative of the receptive field of the model.
Because the model caches the streaming context at different layers, the model
may end up forming indirect dependencies to audio many seconds ago.

|       | full   | cs=32 (1280ms) | 24 (960ms) | 16 (640ms) | 12 (480ms) | 8 (320ms) |
|:-----:|:------:|:------:|:------:|:------:|:------:|:------:|
| full  | 11.00% | -      | -      | -      | -      | -      |
| 16    | -      | -      | -      | 11.70% | 11.84% | 12.14% |
| 8     | -      | -      | 11.50% | 11.72% | 11.88% | 12.28% |
| 4     | -      | 11.40% | 11.53% | 11.81% | 12.03% | 12.64% |
| 2     | -      | 11.46% | 11.67% | 12.03% | 12.43% | 13.25% |
| 1\*\* | -      | 11.59% | 11.85% | 12.39% | 12.93% | 14.13% |

(\*\*: model was never explicitly trained with this setting)

### Inference

Once your model is trained, you need a few manual steps in order to use it with the high-level streaming interfaces (`speechbrain.inference.ASR.StreamingASR`):

1. Create a new directory where you want to store the model.
2. Copy `results/conformer_transducer/<seed>/lm.ckpt` (optional; currently, for streaming rescoring LMs might be unsupported) and `tokenizer.ckpt` to that directory.
3. Copy `results/conformer_transducer/<seed>/save/CKPT+????/model.ckpt` and `normalizer.ckpt` to that directory.
4. Copy your hyperparameters file to that directory. Uncomment the streaming specific keys and remove any training-specific keys. Alternatively, grab the inference hyperparameters YAML for this model from HuggingFace and adapt it to any changes you may have done.
5. You can now instantiate a `StreamingASR` with your model using `StreamingASR.from_hparams("/path/to/model/")`.

The contents of that directory may be uploaded as a HuggingFace model, in which case the model source path can just be specified as `youruser/yourmodel`.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
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
