# Spoken Dialogue State Tracking

This folder contains the scripts to fine-tune cascade and/or end-to-end spoken dialogue state tracking models using the spoken MultiWoz dataset. This recipe considers a T5 Encoder-Decoder backbone model for the textual part and Wav2Vec or Whisper to encode the audio turns. It can be adapted to other models.

## Dataset

The spoken MultiWoz dataset, adapted from MultiWoz 2.1, is available on the [Speech Aware Dialogue System Technology Challenge website](https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html). It consists of audio recordings of user turns and mappings to integrate them with the written agent turns in order to form a full dialogue. Please place all the files in a common folder and rename the manifest files `[split]_manifest.txt` and the split folders `DSTC11_[split]_[vocalization]/`.

- Training data (TTS):
    - [train.tts-verbatim.2022-07-27.zip](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.zip) contains 4 subdirectories, one for each TTS speaker (tpa, tpb, tpc, tpd), and each subdirectories contains all the 8434 dialogs corresponding to the original training set. The TTS outputs were generated using speakers that are available via Google Cloud Speech API.
    - [train.tts-verbatim.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.txt) contains the original dialog training data, which is used to generate the TTS outputs.
- Dev data (TTS):
    - [dev-dstc11.tts-verbatim.2022-07-27.zip](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.tts-verbatim.2022-07-27.zip) contains all the 1000 dialogs corresponding to TTS output from a held-out speaker.
    - [dev-dstc11.2022-07-27.txt](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-07-27.txt) contains the mapping from user utterances back to the original dialog.
- Dev data (Human):
    - [dev-dstc11.human-verbatim.2022-09-29.zip](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.human-verbatim.2022-09-29.zip) contains all the 1000 dialogs turns spoken by crowd workers.
- Test data (TTS):
    - [test-dstc11-tts-verbatim.2022-09-21.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11-tts-verbatim.2022-09-21.zip) contains all the 1000 dialogs corresponding to TTS output from a held-out speaker.
    - [test-dstc11.2022-09-21.txt](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.2022-09-21.txt) contains the mapping from user utterances back to the original dialog.
- Test data (Human):
    - [test-dstc11.human-verbatim.2022-09-29.zip](https://storage.googleapis.com/gresearch/dstc11/test-dstc11.human-verbatim.2022-09-29.zip) contains all the 1000 dialogs turns spoken by crowd workers.
- Test data DST annotations:
    - [test-dstc11.2022-1102.gold.json](https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-1102.gold.json) contains the gold DST annotations for the test set.
    - The [test manifest](https://www.dropbox.com/scl/fi/8232druizd7vixaqwh4kj/test_manifest.txt?rlkey=fns9snqxh8zqvew4i1qetu8hb&dl=0) with the DST annotations integrated in the same format as the other splits (see dataset preprocessing steps).

## Pre-requisites

### Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
cd recipes/MultiWOZ/dialogue_state_tracking/
pip install -r extra_requirements.txt
```

### Dataset download & pre-processing

In order to download and extract the dataset for use by this recipe, you may use the following commands:

```bash
cd YOUR_DATA_FOLDER
wget https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.zip
wget https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.txt
mv train.tts-verbatim.2022-07-27.txt train_manifest.txt
unzip train.tts-verbatim.2022-07-27.zip
# Using only one of the synthetic voices
mv train/tpa DSTC11_train_tts
wget https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.tts-verbatim.2022-07-27.zip
wget https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.human-verbatim.2022-09-29.zip
wget https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-07-27.txt
mv dev-dstc11.2022-07-27.txt dev_manifest.txt
unzip dev-dstc11.tts-verbatim.2022-07-27.zip
mv dev-dstc11.tts-verbatim DSTC11_dev_tts
unzip dev-dstc11.human-verbatim.2022-09-29.zip
mv dev-dstc11.human-verbatim DSTC11_dev_human
wget https://storage.googleapis.com/gresearch/dstc11/test-dstc11-tts-verbatim.2022-09-21.zip
wget https://storage.googleapis.com/gresearch/dstc11/test-dstc11.human-verbatim.2022-09-29.zip
wget 'https://www.dropbox.com/scl/fi/8232druizd7vixaqwh4kj/test_manifest.txt?rlkey=fns9snqxh8zqvew4i1qetu8hb&dl=0' -O test_manifest.txt
unzip test-dstc11-tts-verbatim.2022-09-21.zip
mv tmp/tts DSTC11_test_tts
rm -d tmp
unzip test-dstc11.human-verbatim.2022-09-29.zip
mv test-dstc11.human-verbatim DSTC11_test_human
```

Then you may extract the audio from each split with:

```
python ../meta/extract_audio.py --data_folder YOUR_DATA_FOLDER
```

## How to run

Select the version you want to run with the argument `--version (cascade[_MODEL]|e2e)`. The `cascade_MODEL` approach will consider the transcriptions from the manifest named `SPLIT_manifest_MODEL.txt` in the provided data folder.

### Training

Run the training with:

```
python train.py hparams/train_multiwoz[_with_whisper_enc].yaml --data_folder YOUR_DATA_FOLDER
```

By default the model is trained for 10 epochs with  warmup corresponding to 20% of the training steps. For E2E models we encourage to override the default hyper-parameters with ` --number_of_epochs 20 --warmup_steps 1773` in order to increase the number of epochs while preserving the same scheduling for the first 10 epochs.

### Inference

Run inference with:

```
python train.py hparams/train_multiwoz[_with_whisper_enc].yaml --data_folder YOUR_DATA_FOLDER --inference True
```

Simply add the argument `--gold_previous_state False` to perform the inference with the previously predicted dialogue states.

### Evaluation

Evaluate your predictions in terms of Joint-Goal Accuracy (at turn and dialogue level) and Slot Precision (per slot groups) with the script [evaluate_multiwoz_dst.py](../meta/evaluate_multiwoz_dst.py).

```
python evaluate_multiwoz_dst.py --reference_manifest PATH_TO_SPLIT_MANIFEST --predictions PATH_TO_PREDICTIONS_CSV
```

To evaluate the 95% confidence intervals of the JGA scores, with a bootstrapping strategy, add the argument `--evaluate_ci`.

## Results

### Gold Previous State

|                          | Dev TTS | Dev Human | Test TTS | Test Human |
|:------------------------:|:-------:|:---------:|:--------:|:----------:|
| Cascade (WavLM)          |    58.2 |      55.0 |     57.2 |       53.5 |
| Cascade (Whisper)        |    63.7 |      63.6 |     64.4 |       62.3 |
| Global (WavLM)           |    56.4 |      54.0 |     53.4 |       53.0 |
| Global (Whisper)         |    59.0 |      56.9 |     58.3 |       56.6 |

### Predicted Previous State

|                          | Dev TTS | Dev Human | Test TTS | Test Human |
|:------------------------:|:-------:|:---------:|:--------:|:----------:|
| Cascade (WavLM)          |    19.5 |      16.2 |     17.6 |       15.3 |
| Cascade (Whisper)        |    24.0 |      21.9 |     23.1 |       21.3 |
| Global (WavLM)           |    15.1 |      14.4 |     13.7 |       14.6 |
| Global (Whisper)         |    19.1 |      17.6 |     18.5 |       16.6 |


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
