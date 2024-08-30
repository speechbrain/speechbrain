# BEST-RQ pretraining with SpeechBrain

This folder contains the scripts to train a BEST-RQ model using LibriSpeech. It can be adapted to any dataset as long as you provide the csv or json files. No other adaptation will be required apart from controlling the sequence length and Dynamic Batching arguments to avoid out of memory issues.

More information on the architecture can be found in [the original paper](https://arxiv.org/pdf/2202.01855.).

# Go !
Simply type:
```shell
python train.py hparams/BEST-RQ.yaml --find_unused_parameters
```

Do not forget to replace the `!PLACEHOLDER` variables in the yaml corresponding to your local configuration.

# Pretrained models and results

 To be added

# Tips and tricks
We found that the following parameters can greatly affect downstream performance:
- Batch size
- learning rate (`lr`)
- mask probability (`mask_prob`)

**A note on batch and model size:**

The total batch size is the duration_per_minibatch * nb_gpu * grad_accumulation_factor.
For example, with the recipe given running on 8 GPU will result is about 13 min (100 sec * 8 GPUs * 1 grad_accumulation_factor).

Note that this is MUCH smaller than other models such as wav2vec 2.0 (1.6 hours) the original paper, which has a batch size of around 18 hours. Also, compare to the original paper this implementation uses only 12 conformer layers instead of 24 (this can be easily adjusted by adjusting the `num_encoder_layers` in the hparams file)

We leave the batch and model size small to be able to fit on smaller GPUs and train faster, but better performance will most likely be obtained by increasing the batch and/or model size.
