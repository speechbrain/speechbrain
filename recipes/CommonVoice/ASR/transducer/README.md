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
| French | 2024-03-22 | No | 3.51 | 10.30 | 4.64 | 12.47 | [model](https://www.dropbox.com/scl/fo/kue72ik3vc55xu6u8zjr7/h?rlkey=ie98ktqf9gbunn4x9i3pskedq&dl=0) | 4xV100 32GB |
| Italian | 2024-03-22 | No | 2.47 | 8.49 | 2.69 | 8.92 | [model](https://www.dropbox.com/scl/fo/uyqfo3kwcpkaq26au2foj/h?rlkey=gxlj7xn6bnhjfb5jds8p80fe6&dl=0) | 4xV100 32GB |

The output folders with checkpoints and logs can be found [here](https://www.dropbox.com/sh/852eq7pbt6d65ai/AACv4wAzk1pWbDo4fjVKLICYa?dl=0).

## Streaming model

### WER vs chunk size & left context

The following matrix presents the Word Error Rate (WER%) achieved on CommonVoice
`test` with various chunk sizes (in ms).

The relative difference is not trivial to interpret, because we are not testing
against a continuous stream of speech, but rather against utterances of various
lengths. This tends to bias results in favor of larger chunk sizes.

The chunk size might not accurately represent expected latency due to slight
padding differences in streaming contexts.

The left chunk size is not representative of the receptive field of the model.
Because the model caches the streaming context at different layers, the model
may end up forming indirect dependencies to audio many seconds ago.

|       | full | cs=32 (1280ms) | 16 (640ms) | 8 (320ms) |
|:-----:|:----:|:-----:|:-----:|:-----:|
| it full  | 8.92 | -     | -     |  -   |
| it lc=32    | -    | 10.04 | 10.82 | 12.01 |
| fr full  | 12.47 | -     | -     |  -   |
| fr lc=32    | -    | 13.92 | 14.88 | 16.22 |

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
