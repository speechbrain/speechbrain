# VoxPopuli ASR with Transducers
This folder contains scripts necessary to run an ASR experiment with the VoxPopuli dataset;
Before running this recipe, make sure numba is installed (pip install numba) for faster training!
You can download VoxPopuli at: https://github.com/facebookresearch/voxpopuli

**We only report results for english but you simply need to download a different set to train with a different language!**

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

# VoxPopuli non-streaming results

Results are reported with beam search but without any language model. Models are
trained with dynamic chunk training but decoding is offline.


| Language | Hyperparams file | Train precision | Dev-clean Greedy | Test-clean Greedy | Model link | GPUs |
|:-------------:|:---------------------------:|:-:| :------:| :-----------:| :------------------:| :------------------:|
| English | conformer_transducer.yaml `streaming: True` | fp16 | 9.80 | 10.18 | [Model link](https://www.dropbox.com/scl/fo/y2if76ut4xur5rg9sszj3/h?rlkey=y8wmip8bd06cb82vm2cvmfaz3&dl=0) |6x A40|


# VoxPopuli streaming results

### WER vs chunk size & left context

The following matrix presents the Word Error Rate (WER%) achieved on the test set with various chunk sizes (in ms).

This is with greedy decoding only.


|       | full | cs=32 (1280ms) | 16 (640ms) | 8 (320ms) |
|:-----:|:----:|:-----:|:-----:|:-----:|
| full  | 10.18| -     | -     | -     |
| lc=32 | -    | 10.88 | 11.39 | 12.37 |

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
