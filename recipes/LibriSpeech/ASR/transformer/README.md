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

| Release | hyperparams file | Dev Clean WER (No LM, small beam) | Test Clean WER (Transformer LM) | Test Other WER (Transformer LM) | HuggingFace link | Model link | GPUs |
|:-------------:|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 23-05-23 | branchformer_large.yaml | 2.72 (1.9 with LM) | 2.04 | 4.13 | Not Avail. | [GoogleDrive](https://www.dropbox.com/sh/gxkye4efa6hvl2c/AADO85EkkfbIGe5KjBAU6BrEa?dl=0) | 4xA100 80GB |
| 23-05-23 | conformer_large.yaml | 2.62 (1.9 with LM) | 2.01 | 4.52 | [HuggingFace](https://huggingface.co/speechbrain/asr-conformer-transformerlm-librispeech) | [GoogleDrive](https://www.dropbox.com/sh/ef3chrau8i45ip1/AAD9un8oabOB1a9OiSomZEhZa?dl=0) | 4xA100 80GB |
| 24-03-22 | transformer.yaml | 3.32 | 2.27 | 5.53 | [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | [GoogleDrive](https://www.dropbox.com/sh/653kq8h2k87md4p/AAByAaAryXtQKpRzYtzV9ih5a?dl=0) | 4xV100 32GB |
| 24-03-22 | conformer_small.yaml | 4.05 | 2.49 | 6.1 (**only 13.3M parameters**) | [HuggingFace](https://huggingface.co/speechbrain/asr-conformersmall-transformerlm-librispeech) | [DropBox](https://www.dropbox.com/sh/s0x6ni124858b8i/AAALaCH6sGTMRUVTjh8Tm8Jwa?dl=0) | 1xV100 32GB |
| 06-12-23 | train_hf_whisper.yaml | 3.60 | Not Avail. | Not Avail. | Not Avail. | Not Avail. | 1xA100 40GB |
| 27-03-23 | hyperconformer_8M.yaml | 4.69 | 2.55 | 6.61 (**only 7.9M parameters**) | NA |  [DropBox](https://www.dropbox.com/sh/8jc96avmivr8fke/AABrFEhtWy_3-Q7BHhkh0enwa?dl=0) | 1xP40 24GB
| 27-03-23 | hyperconformer_22M.yaml | 3.19 | 2.23 | 5.54  (**only 21.7M parameters**)  | NA | [DropBox](https://www.dropbox.com/sh/30xsmqj13jexzoh/AACvZNtX1Fsr0Wa1Z3C9rHLXa?dl=0) | 1xP40 24GB
| 03-09-23 | hyperbranchformer_13M.yaml | NA | 2.54 | 6.58  | NA | soon | 1xP40 24GB
| 03-09-23 | hyperbranchformer_25M.yaml | NA | 2.36 | 5.89 | NA | soon | 1xP40 24GB

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
