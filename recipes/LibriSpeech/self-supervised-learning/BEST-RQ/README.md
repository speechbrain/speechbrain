# BEST-RQ streaming and offline pretraining with SpeechBrain

This folder contains the scripts to train a BEST-RQ model using LibriSpeech. It can be adapted to any dataset as long as you provide the csv or json files. No other adaptation will be required apart from controlling the sequence length and Dynamic Batching arguments to avoid out of memory issues.

More information on the architecture can be found in [the original paper](https://arxiv.org/pdf/2202.01855.).

# Go !
Simply type:
```shell
# single GPU example
python train.py hparams/BEST-RQ.yaml --data_folder /path/to/LibriSpeech/ --streaming True

# single node multi GPU example
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=2 train.py hparams/BEST-RQ.yaml --data_folder /path/to/LibriSpeech/ --streaming True
```

Do not forget to replace the `!PLACEHOLDER` variables in the yaml corresponding to your local configuration.

# Tips and tricks
We found that the following parameters can greatly affect downstream performance:
- Batch size (the bigger the better depending on the dataset of interest)
- learning rate (`lr`, depending on the batch size)
- mask probability (`mask_prob` which may need to be adapted to the audio source)

# Finetuning after pretraining
For speech recognition finetuning, simply head to the [ASR / CTC](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/CTC) recipe and use train_with_bestrq.py! Numbers should be equivalent to the paper.

