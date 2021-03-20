# Google Speech Command v0.02 Dataset
This folder contains recipes for command recognition with [Google Speech Command Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands).
The recipes supports 12 or 35 commands.  To run it, please type:

```
python train.py hparams/train_ecapa_tdnn.yaml
```

# Performance summary

[Command accuracy on Google Speech Commands]
| System | Accuracy |
|----------------- | ------------ |
| ECAPA-TDNN v12 | 98.7% |
| ECAPA-TDNN v35 | 96.9% |

You can find the output folder (model, logs, etc) here:
add_link
