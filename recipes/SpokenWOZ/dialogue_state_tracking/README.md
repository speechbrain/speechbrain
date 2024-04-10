# Spoken Dialogue State Tracking

This folder contains the scripts to fine-tune cascade and/or end-to-end spoken dialogue state tracking models using the SpokenWOZ dataset. This recipe considers a T5 Encoder-Decoder backbone model for the textual part and Wav2Vec or Whisper to encode the audio turns. It can be adapted to other models.

## Dataset

The SpokenWOZ dataset is available on their [official website](https://spokenwoz.github.io/SpokenWOZ-github.io/). It consists of human-human task-oriented dialogue recordings associated with Dialogue States for each agent dialogue turn.

- Train and Dev data:
    - [audio_5700_train_dev.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_train_dev.tar.gz) contains each dialogue's audio recording.
    - [text_5700_train_dev.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_train_dev.tar.gz) contains the list of dialogues to consider for the dev split in `valListFile.json` and the annotations for each dialogue turn in `data.json`.
- Test data:
    - [audio_5700_test.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_test.tar.gz) contains each dialogue's audio recording.
    - [text_5700_test.tar.gz](https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_test.tar.gz) contains the annotations for each dialogue turn in `data.json`.

## Pre-requisites

### Dataset download & pre-processing

In order to download and extract the dataset for use by this recipe, you may use the following commands:

```bash
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_train_dev.tar.gz
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_train_dev.tar.gz
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_test.tar.gz
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_test.tar.gz
tar -xzf audio_5700_train_dev.tar.gz --exclude '\._*'
tar -xzf text_5700_train_dev.tar.gz --exclude '\._*'
tar -xzf audio_5700_test.tar.gz --exclude '\._*'
tar -xzf text_5700_test.tar.gz --exclude '\._*'
```

## How to run

Select the version you want to run with the argument `--version (cascade[_MODEL]|e2e)`. The `cascade_MODEL` approach will consider the transcriptions from the manifest named `data_MODEL.json` in the data folder and consider the turn's field `[MODEL]` as the turn's transcription.

### Training

Run the training with:

```
python train.py hparams/train_spokenwoz[_with_whisper_enc].yaml --data_folder YOUR_DATA_FOLDER
```

By default the model is trained for 10 epochs with  warmup corresponding to 20% of the training steps. For E2E models we encourage to override the default hyper-parameters with ` --number_of_epochs 20 --warmup_steps 2328` in order to increase the number of epochs while preserving the same scheduling for the first 10 epochs.

### Inference

Run inference with:

```
python train.py hparams/train_spokenwoz[_with_whisper_enc].yaml --data_folder YOUR_DATA_FOLDER --inference True
```

Simply add the argument `--gold_previous_state False` to perform the inference with the previously predicted dialogue states.

### Evaluation

Evaluate your predictions in terms of Joint-Goal Accuracy (at turn and dialogue level) and Slot Precision (per slot groups) with the script [evaluate_spokenwoz_dst.py](../meta/evaluate_spokenwoz_dst.py).

```
python evaluate_spokenwoz_dst.py --reference_manifest PATH_TO_SPLIT_DATA --predictions PATH_TO_PREDICTIONS_CSV
```

To evaluate the 95% confidence intervals of the JGA scores, with a bootstrapping strategy, add the argument `--evaluate_ci`.

## Results

See [issue](https://github.com/AlibabaResearch/DAMO-ConvAI/issues/87) for misaligned audio files in the test split. The audio was ignored when not available.

|   Gold Previous State    |     Dev     |     Test     |
|:------------------------:|:-----------:|:------------:|
| Cascade (their ASR)      |     82.3    |     63.0     |
| Cascade (Whisper)        |     80.7    |     64.2     |
| Global (WavLM)           |     70.7    |     61.8     |
| Global (Whisper)         |     81.6    |     80.5     |

| Predicted Previous State |     Dev     |     Test     |
|:------------------------:|:-----------:|:------------:|
| Cascade (their ASR)      |     24.6    |     23.4     |
| Cascade (Whisper)        |     24.3    |     23.5     |
| Global (WavLM)           |     22.2    |     20.3     |
| Global (Whisper)         |     26.5    |     24.1     |


## **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

### **Citing**
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
