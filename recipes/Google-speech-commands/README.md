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

You can find the output folder (model, logs, etc) here:
https://drive.google.com/drive/folders/1yPcXVHtrnNM0RhA_IGo8iAdezYZfoViQ?usp=sharing
