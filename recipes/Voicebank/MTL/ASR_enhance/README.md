# Multi-Task (Enhancement and ASR) Recipe

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
> python experiment.py hyperparams/pretrain_perceptual.yaml
> python experiment.py hyperparams/enhance_mimic.yaml
> python experiment.py hyperparams/robust_asr.yaml
```

One important note is that each step depends on one or more pretrained
models, so ensuring these exist and the paths are correct is an
important step. The path in `hyperparams/enhance_mimic.yaml` should
point at the `src_embedding.ckpt` model trained in step 1, and
the path in `hyperparams/enhance_mimic.yaml` should point at
the `enhance_model.ckpt` model trained in step 2.

Joint training can be achieved by adding the `enhance_model` to
the "unfrozen" models so that the weights are allowed to update.
To see enhancement scores, add an enhancement loss after training
is complete and run the script again.

## Latest Results

|-------|---------------|------|-------|---------|----------|
| Input | Mask Loss     | PESQ | eSTOI | Dev WER | Test WER |
|-------|---------------|:----:|:-----:|:-------:|:--------:|
| Clean | (clean phase) | 4.50 | 100.  | xxx     | xxx      |
| Clean | (noisy phase) | 3.85 | 94.6  | xxx     | xxx      |
| Noisy | -             | 1.97 | 78.7  | 4.41    | 3.72     |
| *Joint Training*                                          |
| Noisy | MSE           | xxxx | xxxx  | xxx     | xxx      |
| Noisy | MSE + mimic   | xxxx | xxxx  | xxx     | xxx      |
| *Frozen Mask Training*                                    |
| Noisy | MSE           | 2.72 | 84.8  | 3.95    | 3.46     |
| Noisy | MSE + mimic   | 2.83 | 85.2  | 3.63    | 3.46     |
|-------|---------------|------|-------|---------|----------|
