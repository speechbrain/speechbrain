# Interpretability - ESC50 Dataset

![image](https://github.com/ycemsubakan/speechbrain-1/assets/16886998/8199f0fb-66ee-4f5a-87ee-349695f7e982)

The objective of interpretability is to offer an explanation regarding the decision made by a classifier.

**Post-hoc** interpretation methods aim to build an auxiliary module -- the **interpreter** -- that generates an additional signal in its output
helping the user to better understand why a specific prediction was made by a pre-trained classifier.
You can find some examples [here](https://piqinter.github.io).

Conversely, **by-design** interpretation methods aim to build an interpretable classifier directly from the data.

This recipe implements a number of interpretation techniques.

They utilize pre-trained models obtained from `ESC50/classification`, some of which are readily available in
our HuggingFace repository (e.g., CNN14, Conv2D).

You can train your own classifier by following the instructions provided in the reference readme under `ESC50/classification`.

---------------------------------------------------------------------------------------------------------

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies.

To do this, simply run the following command in your terminal:

```shell
pip install -r extra_requirements.txt
```

---------------------------------------------------------------------------------------------------------

## Supported Methods

### Posthoc Interpretation via Quantization (PIQ)

PIQ utilizes vector quantization on the classifier's representations to reconstruct predictions.

For more details, refer to our [PIQ paper](https://arxiv.org/abs/2303.12659). You can also find samples on the [companion website](https://piqinter.github.io).

To train PIQ on a convolutional classifier using the ESC50 dataset, use the `train_piq.py` script. Run the following command:

```shell
python train_piq.py hparams/piq.yaml --data_folder=/yourpath/ESC50
```

---------------------------------------------------------------------------------------------------------

### Listen to Interpret (L2I)

L2I employs Non-Negative Matrix Factorization to reconstruct the classifier's hidden representation and generate an interpretation audio signal for the classifier decision.

Read more about L2I in the [L2I paper](https://arxiv.org/abs/2202.11479v2).

To train an NMF model on the ESC50 dataset, use the `train_nmf.py` script. Run the command below:

```shell
python train_nmf.py hparams/nmf.yaml --data_folder /yourpath/ESC50
```

Additionally, we provide an L2I interpretation method for a convolutional classifier. To train this method on the ESC50 dataset, use the following command:

```shell
python train_l2i.py hparams/l2i_conv2d.yaml --data_folder /yourpath/ESC50
```

Lastly, we offer the training script for the L2I interpretation method on CNN14. To run this, execute the following command:

```shell
python train_l2i.py hparams/l2i_cnn14.yaml --data_folder /yourpath/ESC50
```

---------------------------------------------------------------------------------------------------------

### Activation Map Thresholding (AMT)

This method interprets the norm of the activation maps as a measure of importance of each input location to the prediction.

We obtain an interpretation mask by thresholding these saliency maps at the q-th quantile.
Hence, the quality of the generated interpretation depends on how interpretable the activation maps are.

Two neural network architectures are currently supported for this method: [FocalNet](https://arxiv.org/abs/2203.11926) and [ViT](https://arxiv.org/abs/2010.11929).
In particular, FocalNet offers a neural network architecture that is interpretable by design.

For more details, refer to our [FocalNet paper](https://arxiv.org/abs/2402.02754).

To generate interpretations for the pre-trained FocalNet or ViT classifiers available on HuggingFace, use the following command:

```shell
python interpret_amt.py hparams/amt_focalnet.yaml --data_folder /yourpath/ESC50
python interpret_amt.py hparams/amt_vit.yaml --data_folder /yourpath/ESC50
```

Alternatively, you can train your own FocalNet or ViT classifiers using the classification recipe under `ESC50/classification`,
and set the corresponding paths as command line arguments or directly in the configuration file. For example:

```yaml
embedding_model_path: ../classification/results/focalnet-base-esc50/1234/save/CKPT+2024-02-08+18-59-37+00/embedding_model.ckpt
classifier_model_path: ../classification/results/focalnet-base-esc50/1234/save/CKPT+2024-02-08+18-59-37+00/classifier.ckpt
```

---------------------------------------------------------------------------------------------------------

## Results

| Hyperparams file  | Fidelity-to-input |  Faithfulness   |   Training time    |                   HuggingFace link                    |                                                         Model link                                                         |    GPUs     |
|:-----------------:|:-----------------:|:---------------:|:------------------:|:-----------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|:-----------:|
| amt_focalnet.yaml |       0.305       |     0.0111      |         -          |                           -                           | [model](https://www.dropbox.com/scl/fo/0hheboei1b35mlrhwj6mt/AOeCdNstN3h8UqFxv0abT7M?rlkey=kx0d1t5v5hqawqwr5ir9weihq&dl=0) | 1xV100 32GB |
|   amt_vit.yaml    |       0.225       |     0.0109      |         -          |                           -                           | [model](https://www.dropbox.com/scl/fo/vlluiqiirlprl3oa7h4sj/APrEFgcIiWjdQhDUEZuNook?rlkey=bhswfspzklypu7k8ndh8lm3st&dl=0) | 1xV100 32GB |
|  l2i_cnn14.yaml   |   Not available   |  Not available  |    25 s / epoch    |                     Not available                     |                     [model](https://www.dropbox.com/sh/cli2gm8nb4bthow/AAAKnzU0c80s_Rm7wx4i_Orza?dl=0)                     |  RTX 3090   |
|  l2i_conv2d.yaml  |   Not available   |  Not available  |  1 min 10 s /epoch |                     Not available                     |                     [model](https://www.dropbox.com/sh/gcpk9jye9ka08n0/AAB-m10r1YEH0rJdUMrCwizUa?dl=0)                     |  RTX 3090   |
|     nmf.yaml      |         -         |        -        |    45 s / epoch    |                     Not available                     |                     [model](https://www.dropbox.com/sh/01exv8dt3k6l1kk/AADuKmikAPwMw5wlulojd5Ira?dl=0)                     |  RTX 3090   |
|     piq.yaml      |   Not available   |   Not available | 1 min 10 s /epoch  | [model](https://huggingface.co/speechbrain/PIQ-ESC50) |                     [model](https://www.dropbox.com/sh/v1x5ks9t67ftysp/AABo494rDElHTiTpKR_6PP_ua?dl=0)                     |  RTX 3090   |
| piq_focalnet.yaml |       0.278       |     0.0111      |   8 min / epoch    |                     Not available                     | [model](https://www.dropbox.com/scl/fo/6mvxb32f0g1i8b4lkdjoq/AGD1xNF8Of2_IXeEsbpXtQE?rlkey=llefue4rxalqyqwxqtwrn8qii&dl=0) | 1xV100 32GB |
|   piq_vit.yaml    |       0.110       |     0.0121      |   5 min / epoch    |                     Not available                     | [model](https://www.dropbox.com/scl/fo/nz4lqwumgz03nanmf9xai/AI21fGwSOzsVvyegTJUEtz4?rlkey=40yjchqgkhcrhbxsa30m3rr6w&dl=0) | 1xV100 32GB |

---------------------------------------------------------------------------------------------------------

## How to Run on Test Sets Only

If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
python train_<method>.py hparams/<method-config>.yaml --data_folder /yourpath/ESC50 --test_only
```

---------------------------------------------------------------------------------------------------------

## Notes

- The recipe automatically downloads the ESC50 dataset. You only need to specify the path to which you would like to download it.

- All the necessary models are downloaded automatically for each training script.

---------------------------------------------------------------------------------------------------------

## Citing

Please cite our [PIQ paper](https://arxiv.org/abs/2303.12659) if you use it in your research:

```bibtex
@misc{paissan2023posthoc,
    title={Posthoc Interpretation via Quantization},
    author={Francesco Paissan and Cem Subakan and Mirco Ravanelli},
    year={2023},
    eprint={2303.12659},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

Please cite our [FocalNet paper](https://arxiv.org/abs/2402.02754) if you use it in your research:

```bibtex
@inproceedings{dellalibera2024focal,
    title={Focal Modulation Networks for Interpretable Sound Classification},
    author={Luca Della Libera and Cem Subakan and Mirco Ravanelli},
    booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) XAI-SA Workshop},
    year={2024},
}
```

If you use **SpeechBrain**, please cite:

```bibtex
@misc{speechbrain,
    title={{SpeechBrain}: A General-Purpose Speech Toolkit},
    author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
    year={2021},
    eprint={2106.04624},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    note={arXiv:2106.04624}
}
```

---------------------------------------------------------------------------------------------------------

## About SpeechBrain

- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

---------------------------------------------------------------------------------------------------------
