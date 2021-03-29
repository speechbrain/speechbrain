# Voicebank Multi-Task (Enhancement and ASR) Recipe

This recipe combines enhancement and ASR to improve performance on both tasks.
The technique we use in this recipe is called _mimic loss_ [1, 2, 3] and
is performed in three main stages:

1. Pretrain an acoustic model as a perceptual model of speech, used to
   judge the perceptual quality of the outputs of the enhancement model.
2. Train an enhancement model by freezing the perceptual model, passing
   clean and enhanced features to the perceptual model, and generating
   a loss using the MSE between the outputs of the perceptual model.
3. Freezing the enhancement model and training a robust ASR model
   to recognize the enhanced outputs.

This approach is similar to joint training of enhancement and ASR models,
but maintains the advantages of interpretability and independence, since
each model can be used for other data or tasks without requiring the
co-trained model.

To train these models from scratch, you can run these three steps
using the following commands:

```
> python train.py hparams/pretrain_perceptual.yaml
> python train.py hparams/enhance_mimic.yaml
> python train.py hparams/robust_asr.yaml
```

One important note is that each step depends on one or more pretrained
models, so ensuring these exist and the paths are correct is an
important step. The path in `hparams/enhance_mimic.yaml` should
point at the `src_embedding.ckpt` model trained in step 1, and
the path in `hparams/enhance_mimic.yaml` should point at
the `enhance_model.ckpt` model trained in step 2.

Joint training can be achieved by adding the `enhance_model` to
the "unfrozen" models so that the weights are allowed to update.
To see enhancement scores, add an enhancement loss after training
is complete and run the script again.

## Latest Results

The PESQ and eSTOI results are generated using the test set, and the
WER results are generated over 3 runs.
The last 5 epochs are combined so no validation
data is used to choose checkpoints.

All but the last results are generated using the very efficient
CNN + Transformer model. The last results are generated with CRDNN
which is slower, but a bit better.

| Input | Mask Loss     | PESQ | eSTOI | dev WER | tst WER  |
|-------|---------------|:----:|:-----:|:-------:|:--------:|
| Clean | (clean phase) | 4.50 | 100.  | 1.44    | 2.29     |
| Clean | (noisy phase) | 3.85 | 94.6  | 1.26    | 2.45     |
| Noisy | -             | 1.97 | 78.7  | 4.19    | 3.46     |
| *Joint Training*                                          |
| Noisy | MSE           | 2.45 | 83.3  | 3.40    | 3.12     |
| Noisy | MSE + mimic   | 2.58 | 83.5  | 3.50    | 3.32     |
| *Frozen Mask Training*                                    |
| Noisy | MSE           | 2.72 | 84.8  | 3.48    | 3.12     |
| Noisy | MSE + mimic   | 2.87 | 85.2  | 3.20    | 2.96     |
| *CRDNN model*                                             |
| Noisy | MSE           | 2.74 | 85.4  | -       | -        |
| Noisy | MSE + mimic   | 2.99 | 86.4  | -       | -        |

## References

[1] Deblin Bagchi, Peter Plantinga, Adam Stiff, Eric Fosler-Lussier, “Spectral Feature Mapping with Mimic Loss for Robust Speech Recognition.” ICASSP 2018 [https://arxiv.org/abs/1803.09816](https://arxiv.org/abs/1803.09816)

[2] Peter Plantinga, Deblin Bagchi, Eric Fosler-Lussier, “An Exploration of Mimic Architectures for Residual Network Based Spectral Mapping.” SLT 2018 [https://arxiv.org/abs/1809.09756](https://arxiv.org/abs/1809.09756)

[3] Peter Plantinga, Deblin Bagchi, Eric Fosler-Lussier, “Phonetic Feedback For Speech Enhancement With and Without Parallel Speech Data.” ICASSP 2020 [https://arxiv.org/abs/2003.01769](https://arxiv.org/abs/2003.01769)
