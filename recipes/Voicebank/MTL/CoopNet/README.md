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

    python train.py hparams/train_3layer.yaml --data_folder path/to/noisy-vctk-16k

This recipe makes a 3 layer cooperative network.
The fuse mask is an attention mask.

The models and skip connections are kept in the models/ folder, for clarity.

To train this from scratch, there are other yaml files that can be used

    python train.py hparams/pretrain_asr_and_se.yaml --data_folder path/to/noisy-vctk-16k
    python train.py hparams/pretrain_1layer.yaml --data_folder path/to/noisy-vctk-16k --ckpt_path path/to/pretrained
    python train.py hparams/train_3layer.yaml --data_folder path/to/noisy-vctk-16k # change pretrainer paths

Just make sure to change the pretrainer paths so that they point to the right checkpoints.

Currently, the best performance is the following, with intermediate training points reported:

| Input | Mask Loss           | PESQ | eSTOI | dev PER | tst PER  | Epochs |
|-------|---------------------|:----:|:-----:|:-------:|:--------:|:------:|
| Clean | CTC only            |  -   |   -   | 14.29   |    -     | 10     |
| Noisy | MSE on SE only      | 2.24 | 93.4  |    -    |    -     | 50     |
| *Joint Training using pretrained modules*                                |
| Noisy | MSE + CTC 1 layer   | 2.66 | 84.7  | 19.10   | 18.37    | 25     |
| Noisy | MSE + CTC 2 layers  | 2.75 | 85.1  | 16.32   | 16.01    | 50     |
| Noisy | MSE + CTC 3 layers  | 2.86 | 85.4  | 16.26   | 15.77    | 30     |