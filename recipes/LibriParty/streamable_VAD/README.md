# Voice Activity Detection (VAD) with LibriParty
This folder contains scripts for training a streamable VAD on the [LibriParty dataset](https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing).
LibriParty contains sequences of 1 minute compose of speech sentences (sampled from LibriSpeech) corrupted by noise and reverberation.
Data augmentation with open_rir, musan, CommonLanguge is used as well. Make sure you download all the datasets before staring the experiment:
- LibriParty: https://www.dropbox.com/s/ns63xdwmo1agj3r/LibriParty.tar.gz?dl=1
- Musan: https://www.openslr.org/resources/17/musan.tar.gz
- CommonLanguage: https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1


# Training a RNN-based VAD
Run the following command to train the model:
`python train.py hparams/streamable.yaml --data_folder=your_path/LibriParty/dataset/ --musan_folder=your_path/musan/ --commonlanguage_folder=your_path/common_voice_kpd/`
(change the paths with your local ones)


# Results
| Release | hyperparams file | Test Precision | Test Recall. | Test F-Score | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| -----------:| -----------:|
| 2021-09-09 | streamable.yaml |  0.9417 | 0.9007 | 0.9208 | [Model](https://drive.google.com/drive/folders/1_L8mp1lpnIGEf8SUUNBmZuLSw5aYoY4l?usp=drive_link) | NVIDIA RTX 3090 |

The pre-trained model is available on the [HF Hub](https://huggingface.co/speechbrain/stream-vad-crdnn-libriparty).


# Training Time
About 2 minutes for each epoch with a NVIDIA RTX 3090.

## Environment setup
To setup the environment, run:
```
pip install speechbrain
conda install -c conda-forge 'ffmpeg<7' # needed for streamreader -- inference only

git clone https://github.com/speechbrain/speechbrain/
cd speechbrain/recipes/LibriParty/streamable_VAD/
pip install -r extra-dependencies.txt
```

## Running realtime inference
**Note:** as of now, PyTorch's streamreader only supports Apple devices, and so does our script. We will add support to more in the future.
To run real-time inference, you can download and adapt the [inference script](https://huggingface.co/fpaissan/stream-vad-crdnn-libriparty/blob/main/inference.py).

To download the inference script, run:
```
git clone https://github.com/speechbrain/speechbrain/
```
The inference script is located in `recipes/LibriParty/streamable_VAD/inference.py`.

In order to run the script, you should insert the ID of your microphone, you can do so on your system following the next steps.

To retrieve the ID of your microphone, run:
```ffmpeg -hide_banner -list_devices true -f avfoundation -i dummy```
and copy the ID of the microphone.

After retrieving your device ID, modify the script as follows you can run the inference script with
```
cd speechbrain/recipes/LibriParty/streamable_VAD/
python inference.py {MICROPHONE_ID}
```

This will open a window displaying the raw waveform on the top row, and the speech presence probability on the bottom row. You can close the demo via CTRL+C.
After the execution, the script saves two images containing the processed waveform both offline (offline_processing.png) and realtime (streaming.png) for comparison.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/

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

