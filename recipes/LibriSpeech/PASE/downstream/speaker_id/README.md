## Downstream task - Speaker identification
This downstream task uses pretrained PASE encoder model, adds classification layer on top of it and fine-tunes the encoder model parameters to suit this task.

### Downstream task configuration
- The downstream task is trained on minilibrispeech data.
- Once you have the encoder ckpt file (containing pretrained weights), update the encoder configuration in `train.yaml` to make sure the configs used to get ckpt is the one being used for training downstream task. If the configs used to train the encoder and the config used in this downstream task is different, then the pretrainer cannot load the model.
- The hyperparameters are to be specified in the `train.yaml` before starting to train the model.

### Steps to run the downstream task
1. Update the `pretrained_path` in the `train.yaml` file to the directory containing the encoder's ckpt file. Also, update the filename in `paths` key of the `pretrainer` parameter accordingly.
1. Once done, the downstream task can be trained using `python train.py train.yaml`. This will download the minilibrispeech dataset if not available in the current directory and then run the training.
