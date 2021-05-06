Recipe written by Nicolas Duchêne, Sreeramadas Sai Aravind, and Émile Dimas.

This recipe uses a few extra packages, please install them with

    pip install -r requirements.txt

wandb can be set offline with, which avoids having to interact with the prompt:

    wandb offline

The logger for wandb is configured via `hparams/logger.yaml`. In it, all but the `initializer` key can be used as specified in their documentation: https://docs.wandb.ai/ref/python/init

The `initializer` key is there to avoid to put a dependency on wandb for the `speechbrain.utils.train_logger.py` module.

To remove wandb altogether, please toggle it at the beginning of the main routine in train.py.

To use this recipe, one needs to have the VoiceBank dataset available, and replace the placeholder value in the yaml file for where you have the dataset.

Simply use

    python train.py hparams/train_3layer.yaml

This recipe makes a 3 layer cooperative network.
The fuse mask is an attention mask.

The models and skip connections are kept in the models/ folder, for clarity.

To train this from scratch, there are other yaml files that can be used

    python train.py hparams/pretrain_asr_and_se.yaml
    python train.py hparams/pretrain_1layer.yaml
    python train.py hparams/train_3layer.yaml

Just make sure to change the pretrainer paths in `train_3layer.yaml`.