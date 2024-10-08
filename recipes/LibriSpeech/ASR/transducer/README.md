# LibriSpeech ASR with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the LibriSpeech dataset;
Before running this recipe, make sure numba is installed (pip install numba)
You can download LibriSpeech at http://www.openslr.org/12

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

# Librispeech Results

Dev. clean is evaluated with Greedy Decoding while the test sets are using Greedy Decoding OR a RNNLM + Beam Search.
Evaluation is performed in fp32. However, we found that during inference, fp16 or bf16 autocast has very little incidence on the WER.

| Release | Hyperparams file | Train precision | Dev-clean Greedy | Test-clean Greedy | Test-other Greedy | Test-clean BS+RNNLM | Test-other BS+RNNLM | Model link | GPUs |
|:-------------:|:---------------------------:|:-:| :------:| :-----------:| :------------------:| :------------------:| :------------------:| :--------:| :-----------:|
| 2023-12-12 | conformer_transducer.yaml `streaming: True` | bf16 | 2.56% | 2.72% | 6.47% | \* | \* | [DropBox](https://www.dropbox.com/scl/fo/kl1eikmoauygwqcx8ok4r/AMkreKLzHtxPtqnoXzUerko?rlkey=juk374k210b76lbnblh7or95d&st=1ugwe9e3&dl=0) | [4x A100SXM4 40GB](https://docs.alliancecan.ca/wiki/Narval/en) |

<sub>\*: not evaluated due to performance issues, see [issue #2301](https://github.com/speechbrain/speechbrain/issues/2301)</sub>

## Streaming model

### WER vs chunk size & left context

The following matrix presents the Word Error Rate (WER%) achieved on LibriSpeech
`test-clean` with various chunk sizes (in ms) and left context sizes (in # of
chunks).

The relative difference is not trivial to interpret, because we are not testing
against a continuous stream of speech, but rather against utterances of various
lengths. This tends to bias results in favor of larger chunk sizes.

The chunk size might not accurately represent expected latency due to slight
padding differences in streaming contexts.

The left chunk size is not representative of the receptive field of the model.
Because the model caches the streaming context at different layers, the model
may end up forming indirect dependencies to audio many seconds ago.

|       | full | cs=32 (1280ms) | 24 (960ms) | 16 (640ms) | 12 (480ms) | 8 (320ms) |
|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| full  | 2.72%| -     | -     | -     | -     | -     |
| lc=32 | -    | 3.09% | 3.07% | 3.26% | 3.31% | 3.44% |
| 16    | -    | 3.10% | 3.07% | 3.27% | 3.32% | 3.50% |
| 8     | -    | 3.10% | 3.11% | 3.31% | 3.39% | 3.62% |
| 4     | -    | 3.12% | 3.13% | 3.37% | 3.51% | 3.80% |
| 2     | -    | 3.19% | 3.24% | 3.50% | 3.79% | 4.38% |

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
