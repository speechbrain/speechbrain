# Google Speech Command v0.02 Dataset
This folder contains recipes for command recognition with [Google Speech Command Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).
The recipes supports 12 or 35 commands.  To run it, please type:

```
# commands should exist under data_folder and rirs_noises.zip should under rir_folder
python train_crn.py hparams/resnet.yaml --data_folder=/home/wangwei/diskdata/corpus/kws/mobvoi_hotword_dataset/mobvoi_hotword_dataset --rir_folder /home/wangwei/work/corpus/rir --data_parallel_backend
python train.py hparams/xvect.yaml --data_folder=/path_to_/GSC --seed=1234  --number_of_commands=35 --percentage_unknown=0 --percentage_silence=0 (v35 task)
```

# Performance summary

[Command accuracy on Google Speech Commands]
| System | ErrorRate |
|----------------- | ------------ |
| BIGRU | 1.95e-3 |
| xvector + augment v35 | 97.43% |

You can find the output folder (model, logs, etc) here:
https://drive.google.com/drive/folders/1yPcXVHtrnNM0RhA_IGo8iAdezYZfoViQ?usp=sharing
