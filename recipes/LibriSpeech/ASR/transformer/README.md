# LibriSpeech ASR with Transformers or Whisper models.
This folder contains the scripts to train a Transformer-based speech recognizer or the scripts to fine-tune the Whisper encoder-decoder model.

You can download LibriSpeech at http://www.openslr.org/12

# How to run
```shell
python train_with_whisper.py hparams/train_hf_whisper.yaml
python train.py hparams/transformer.yaml

```

# How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
python train_with_whisper.py hparams/train_hf_whisper.yaml --test_only
python train.py hparams/transformer.yaml --test_only
```

**If using a HuggingFace pre-trained model, please make sure you have "transformers"
installed in your environment (see extra-requirements.txt)**

# Results

## Whisper Finetuning Result:

Following table contains whisper-finetuning results for 1 epoch using Whisper model, freezing encoder and finetuning decoder.
| Release | Model | commit hash | hyperparams file | LM | Dev Clean WER | Test Clean WER | Test Other WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:| -----:|-----:|:---------------------------:|  -----:| -----:| -----:|  :-----------: |:-----------:| :-----------:|
| 2024-03-28 | large-v3 | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 2.00% | 1.96% | 4.30% | Not Avail. | [DropBox](https://www.dropbox.com/scl/fo/d3gmgf6q79byuhzozdwz8/AGFQwMWJ5hqB466GXTnL72M?rlkey=gmi157oa36vvo9c9o1z4oys0e&dl=0) |  2xV100S 32GB |
| 2024-03-28 | medium.en | [e4e2e13](https://github.com/speechbrain/speechbrain/pull/2450/commits/e4e2e135e9edafc6a26fc9aa4df9a94eaf86de41) | train_hf_whisper.yaml | No | 2.35% | 2.40% | 5.59% | Not Avail. | [DropBox](https://www.dropbox.com/scl/fo/a233v5q1gjpy4nyfh2gq0/ALCbTe3UwAjfia7XI2GLx7A?rlkey=lnoxdpiyxm6lg461ptbdrifcj&dl=0160) |  2xV100S 32GB |


## Transformers

| Release | hyperparams file | Dev Clean WER (No LM, small beam) | Test Clean WER (Transformer LM) | Test Other WER (Transformer LM) | HuggingFace link | Model link | GPUs |
|:-------------:|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 23-05-23 | branchformer_large.yaml | 2.72 (1.9 with LM) | 2.04 | 4.13 | Not Avail. | [DropBox](https://www.dropbox.com/scl/fo/qhtds5rrdvhhhjywa7ovw/AMiIL5YvQENw5JKVpzXlP5o?rlkey=hz8vlpy3qf9kcyfx0cox089e6&st=ufckv6tb&dl=0) | 4xA100 80GB |
| 23-05-23 | conformer_large.yaml | 2.62 (1.9 with LM) | 2.01 | 4.52 | [HuggingFace](https://huggingface.co/speechbrain/asr-conformer-transformerlm-librispeech) | [DropBox](https://www.dropbox.com/scl/fo/9we244tgdf47ay20hrdoz/AKnoqQ13nLwSv1ITeJEQ3wY?rlkey=05o5jiszr8rhj6dlprw87t2x4&st=u2odesyk&dl=0) | 4xA100 80GB |
| 24-03-22 | transformer.yaml | 3.32 | 2.27 | 5.53 | [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | [DropBox](https://www.dropbox.com/sh/653kq8h2k87md4p/AAByAaAryXtQKpRzYtzV9ih5a?dl=0) | 4xV100 32GB |
| 24-03-22 | conformer_small.yaml | 4.05 | 2.49 | 6.1 (**only 13.3M parameters**) | [HuggingFace](https://huggingface.co/speechbrain/asr-conformersmall-transformerlm-librispeech) | [DropBox](https://www.dropbox.com/sh/s0x6ni124858b8i/AAALaCH6sGTMRUVTjh8Tm8Jwa?dl=0) | 1xV100 32GB |
| 27-03-23 | hyperconformer_8M.yaml | 4.69 | 2.55 | 6.61 (**only 7.9M parameters**) | Not Avail. |  [DropBox](https://www.dropbox.com/sh/8jc96avmivr8fke/AABrFEhtWy_3-Q7BHhkh0enwa?dl=0) | 1xP40 24GB
| 27-03-23 | hyperconformer_22M.yaml | 3.19 | 2.23 | 5.54  (**only 21.7M parameters**)  | Not Avail. | [DropBox](https://www.dropbox.com/sh/30xsmqj13jexzoh/AACvZNtX1Fsr0Wa1Z3C9rHLXa?dl=0) | 1xP40 24GB
| 03-09-23 | hyperbranchformer_13M.yaml | NA | 2.54 | 6.58  | Not Avail. | Not Avail. | 1xP40 24GB
| 03-09-23 | hyperbranchformer_25M.yaml | NA | 2.36 | 5.89 | Not Avail. | Not Avail. | 1xP40 24GB
| 05-01-24 | bayesspeech.yaml | 4.28 | 2.84 | 6.27 | Not Avail. | [DropBox](https://www.dropbox.com/scl/fo/cdken4jqfj96ev1v84jxm/h?rlkey=25eu1ytgm5ac51zqj8p65zwxd&dl=0) | 1xV100 32GB |
| 07-06-24 | mwmha_transformer_small.yaml | 4.60 | 2.66 | 6.50 (**only 12.7M parameters**) | NA | NA | 1xA40 48GB |
| 07-06-24 | mwmha_transformer_medium.yaml | 3.55 | 2.26 | 5.66 (**only 39.9M parameters**) | NA | NA | 1xA40 48GB |


# **About HyperConformer**
HyperConformer is a new architecture, which replaces the self-attention mechanism of Conformer with the linear-time token mixing architecture HyperMixer.
It achieves competitive or better results than Conformer while requiring less memory and compute.

- Paper: https://arxiv.org/abs/2305.18281
- HyperMixer code: https://github.com/idiap/hypermixing

Please cite HyperConformer if you use it for your research or business.

```bibtex
@inproceedings{mai23_interspeech,
  author={Florian Mai and Juan Zuluaga-Gomez and Titouan Parcollet and Petr Motlicek},
  title={{HyperConformer}: Multi-head HyperMixer for Efficient Speech Recognition},
  year=2023,
  booktitle={Proc. Interspeech 2023},
  pages={2213--2217},
  doi={10.21437/Interspeech.2023-1611}
}
```

# **About MW-MHA Transformer**
Multi-Window Multi-Head Attention (MW-MHA) is a new Multi-Head attention module where the constituent individual attention heads operate on different local sizes of the input sequence, capturing local-global dependencies more effectively. The method was proposed in the paper "Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners" by Yadav et al. (2024), where it was shown to capture better local-global dependencies when learning general-purpose audio representations.

Here, we simply replaced the standard MHA in tranformer encoder with MW-MHA, which yields substantial improvements in ASR performance.

- Paper: https://openreview.net/forum?id=Q53QLftNkA
- Code: https://github.com/SarthakYadav/mwmae-jax-official

If you use MW-MHA in your work, please cite the following paper:

```bibtex
@inproceedings{
  yadav2024masked,
  title={Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners},
  author={Sarthak Yadav and Sergios Theodoridis and Lars Kai Hansen and Zheng-Hua Tan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Q53QLftNkA}
  }
```

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{ravanelli2024opensourceconversationalaispeechbrain,
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
