# Audio/Speech LLM with LTU-AS.
This folder contains the scripts to train the [LTU-AS](https://arxiv.org/pdf/2309.14405.pdf) model based on [Llama3](https://ai.meta.com/blog/meta-llama-3/). LTU-AS is an instruction-following LLM that can jointly understand audio and speech.

## Open-ASQA dataset
Open-ASQA is a collection of vaious public datasets and the users need to prepare the audios themselves. In our recipe,OPENASQA contains:
 - AudioSet (Audioset_20k, Audioset_2m, AS_strong)
 - VGGSound
 - FSD50K
 - AudioCaps
 - Clotho
 - IEMOCAP
 - LibriTTS
 - Voxceleb2
 - CMU-MOSEI
 - FMA

It should be noted that FreeSound and Sound Bible are not used.

## Installing Extra Dependencies
Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
cd recipes/OPENASQA/ltu-as
pip install -r extra_requirements.txt
```
> **Note**
> “Llama 3 is licensed under the LLAMA 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

## How to run
Before running the training stages, the preparation stage needs to be completed at first by running

```python
# Since training LLM can be too memory-consuming, the whisper-feature extraction is separated from the training in order to save memory

# Data preparation stage: prepare datasets and pre-extract whisper embeddings, then discretise with an average pooling.

python openasqa_prepare.py hparams/prepare_stage_0.yaml
```

The preparation stage takes time. It can be parallelized manually if more gpus are allowed.

For training, a curriculum is adopted to LTU-AS, which is split into 3 stages:

| Stage | config | Task | Audio_Proj | TLTR | LoRA | Total Params |
|:------:|:--------------------:|:----------------:|:-------------:| :-------------:| :-------------:| :----------:|
| 1 | train_stage_1.yaml | Classification | Trainable |  Frozen |  Without | 5.2M |
| 2 | train_stage_2.yaml | Classification | Trainable | Trainable | Trainable | 48.0M |
| 3 | train_stage_3.yaml | All | Trainable | Trainable | Trainable | 48.0M |

As can be seen that only a tiny part of the model's parameters are trainable, compared to the entire model size.

Run the training scripts:
```python
# traning stage 1: only train the audio projection layer because it is radomly initialized, and only on classification tasks
python train.py hparams/train_stage_1.yaml

# traning stage 2: train the audio projection layer + TLTR + LoRA, only on classification tasks
python train.py hparams/train_stage_2.yaml

# traning stage 3: train the audio projection layer + TLTR + LoRA, on all the tasks
python train.py hparams/train_stage_3.yaml
```

It should be noted that stage 2 continues training the models that are trained from stage 1, while stage 3 should continue training those from stage 2. This is implemented via the `Pretrainer` class that can be found in the yaml.

The model is evaluated on five different tasks:
1. Emotion Recognition on IEMOCAP
2. Audio Classification on ESC50 (the ltu-as model outputs an audio description, then an external llm such as llama3-70b is used for classification.)
3. Gender Classfication on Voxceleb2-test
4. Age Prediction on Voxceleb2-test
5. ASR on LibriSpeech test-clean

Run the evaluation:
```python
python evaluate.py hparams/evaluate.yaml
```

## GPU Usage and Training Time
4 * 48G gpus were used to train this model.

Mixed precision was enabled with `fp16`, which can be set to `bf16` depending on the gpus used.

Gradient-checkpointing was enabled in order to save memory but this doubles the training time. If gpus with large enough vram are used, gradiant-checkpointing can be disabled in the yaml by setting `gradient_checkpointing: False` in order to save time.

Training time for each stage with gradient-checkpointing:
| Stage | n_gpus | n_epochs | Total Time |
|:------:|:------:|:------:|:------:|
| 1 | 4 | 2 | 30h |
| 2 | 4 | 2 | 30h |
| 3 | 4 | 1 | 90h |

## Results
The evaluation was carried out on 5 different close-ended tasks:
| model | Emotion Recognition Iemocap (Acc) | ASR Librispeech test-clean (WER) | Audio Classification ESC-50 (Acc) | Age Prediction Voxceleb2-test (MAE) | Gender Classification Voxceleb2-test (F1) |
|:-----------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|
| original model in the paper | 65.2% | 4.9% | 76.6% | 7.3 | 90.8% |
| our model | 69.5% | 1.45% (with Whisper large v3) | 80.8% | 6.67 | 98.8% |


## Pretrained models and Inference
The pretrained models and all the training logs can be found [here](https://www.dropbox.com/scl/fo/dnfdrkb5jl0mk93svl8eo/AHz79M05CX5O3SqxOD8EXgk?rlkey=qu6v0qasa3sxr01rbpnxdep1t&st=we15otgb&dl=0).

A huggingface interface can be found [here](https://huggingface.co/speechbrain/speech-llm-LTU-AS-openasqa).

# **Citing LTU-AS**
```bibtex
@inproceedings{gong_ltuas,
  title={Joint Audio and Speech Understanding},
  author={Gong, Yuan and Liu, Alexander H and Luo, Hongyin, and Karlinsky, Leonid and Glass, James},
  year={2023},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
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
