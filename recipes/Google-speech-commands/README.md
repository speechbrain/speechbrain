# Google Speech Command v0.02 Dataset
This folder contains recipes for command recognition with [Google Speech Command Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).
The recipes supports 12 or 35 commands.  To run it, please type:

```
python train.py hparams/xvect.yaml --data_folder=/path_to_/GSC (V12 task)
python train.py hparams/xvect.yaml --data_folder=/path_to_/GSC --seed=1234  --number_of_commands=35 --percentage_unknown=0 --percentage_silence=0 (v35 task)
```

# Performance summary

[Command accuracy on Google Speech Commands]
| System | Accuracy |
|----------------- | ------------ |
| xvector + augment v12 | 98.14% |
| xvector + augment v35 | 97.43% |



# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/google_speech_command_xvector

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://drive.google.com/drive/folders/1yPcXVHtrnNM0RhA_IGo8iAdezYZfoViQ?usp=sharing


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
