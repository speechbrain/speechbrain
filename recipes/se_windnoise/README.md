# Template for Speech Enhancement

This folder provides a working, well-documented example for training
a speech enhancement model from scratch, based on a few hours of
data. The data we use is from Mini Librispeech + OpenRIR.

There are four files here:

* `train.py`: the main code file, outlines entire training process.
* `hparam/train_gru.yaml`: the hyperparameters file, sets all parameters of execution.
* `model/custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data
    manifests.

To train an enhancement model, just execute the following on the command-line:

```bash
python train.py hparams/train_gru.yaml --data_folder /home/wangwei/work/corpus/asr/MiniLibriSpeech --rir_folder /home/wangwei/work/corpus/RIR --data_parallel_backend

python test.py hparams/train_gru.yaml --data_folder /home/wangwei/work/corpus/asr/MiniLibriSpeech --rir_folder /home/wangwei/work/corpus/RIR --data_parallel_backend
```

- tensorbaord

  ```
  # Launch tensorboard (default port is 6006)
  tensorboard --logdir logs --port 6006
  # Open port-forwarding connection. Add -Nf option not to open remote.
  ssh -L 8008:localhost:6006 wangwei@192.168.1.9
  ```

  

- result

  | model | dev  |      | test |      |
  | ----- | ---- | ---- | ---- | ---- |
  |       | PESQ | STOI | PESQ | STOI |
  | BiGRU | 2.28 | 0.89 | 2.33 | 0.89 |
  | GRU   |      |      |      |      |
