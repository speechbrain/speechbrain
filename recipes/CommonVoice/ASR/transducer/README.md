# CommonVoice ASR with Transducers.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice 14.0 dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/) and pytorch 2.0

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
Enabling half precision can significantly reduce the peak VRAM requirements. For example, in the case of the Conformer Transducer recipe trained with Librispeech, the peak VRAM decreases from 39GB to 12GB when using fp16.
According to our tests, the performance is not affected.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset
with our transducers:
- French
- Italian
- English

# Results (non-streaming)

Results are obtained with beam search and no LM (no-streaming i.e. full context).

| Language | Release |  LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| French | 2024-02-24 | No | 4.25 | 11.81 | 5.23 | 13.64 | [model]() | [model]() | 4xA40 40GB |

The output folders with checkpoints and logs can be found [here](https://www.dropbox.com/sh/852eq7pbt6d65ai/AACv4wAzk1pWbDo4fjVKLICYa?dl=0).

## Streaming model

To be added..

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
