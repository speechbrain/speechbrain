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

The recipe also makes use of the WHAM! noise dataset, which can be downloaded from [here](http://wham.whisper.ai/).

---------------------------------------------------------------------------------------------------------

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies.

To do this, simply run the following command in your terminal:

```shell
pip install -r extra_requirements.txt
```

---------------------------------------------------------------------------------------------------------

## Supported Methods

Some results that are obtained with this recipe on the OOD evaluation are as follows:

|Method | AI    | AD  	| AG  	|FF   	|Fid-In   | SPS | COMP |
|---	|---	|---	|---	|---	| ----    | --   | ---  |
|L-MAC 	| 61.62 | 3.83 | 33.48 | 0.40 | 0.82 | 0.93 | 9.77 |
|L-MAC FT | 58.87 | 4.89 | 30.84 | 0.40 | 0.82 | 0.82 | 10.65 |
|L2I   	| 6.75  |25.93 	|1.25  	|0.26  | 0.01  | 0.58   | 11.38  |

Please, refer to the [L-MAC paper](https://arxiv.org/abs/2403.13086) for more information about the evaluation metrics.


### Listenable Maps for Audio Classifiers (L-MAC)

LMAC trains an interpreter on the classifier's representations to reconstruct interpretations based on a amortized inference loss.

For more details, refer to our [L-MAC paper](https://arxiv.org/abs/2403.13086). You can also find samples on the [companion website](https://francescopaissan.it/lmac/).

To train LMAC on a convolutional classifier using the ESC50 dataset, use the `train_lmac.py` script. Run the following command:

```shell
python train_lmac.py hparams/lmac_cnn14.yaml --data_folder=/yourpath/ESC50
```

Eventually, you can use WHAM! augmentation to boost the interpretations performance, using:
```shell
python train_lmac.py hparams/lmac_cnn14.yaml --data_folder=/yourpath/ESC50 --add_wham_noise True --wham_folder=/yourpath/wham_noise
```
**Note**: The WHAM! noise dataset can be downloaded from [here](http://wham.whisper.ai/).

To run the finetuning stage of the interpreter, use
```shell
python train_lmac.py hparams/lmac_cnn14.yaml --data_folder=/yourpath/ESC50 \
    --add_wham_noise True --wham_folder=/yourpath/wham_noise \
    --finetuning True --pretrained_interpreter=/yourLMACcheckpointpath/psi_model.ckpt --g_w 4
```
where $g_w$ is the guidance weight for the interpreter.

#### Specifying the pretrained classifier

The pretrained classifier to be interpreted is specified with the variables `embedding_model_path`, and `classifier_model_path`. The default model is a model we trained on ESC50, however, if you would like to specify your own model just use paths that point to your own model.

---------------------------------------------------------------------------------------------------------

### Posthoc Interpretation via Quantization (PIQ)

PIQ utilizes vector quantization on the classifier's representations to reconstruct predictions.

For more details, refer to our [PIQ paper](https://arxiv.org/abs/2303.12659). You can also find samples on the [companion website](https://piqinter.github.io).

To train PIQ on a convolutional classifier using the ESC50 dataset, use the `train_piq.py` script. Run the following command:

```shell
python train_piq.py hparams/piq.yaml --data_folder=/yourpath/ESC50
```

Note that the command above runs the recipe for PIQ for a conv2d classifier used in the PIQ paper. Note that we also have yaml files for interpreting a ViT model and a focalnet using PIQ. (respectively, `piq_vit.yaml`, `piq.yaml`.

---------------------------------------------------------------------------------------------------------

### Listen to Interpret (L2I)

L2I employs Non-Negative Matrix Factorization to reconstruct the classifier's hidden representation and generate an interpretation audio signal for the classifier decision.

Read more about L2I in the [L2I paper](https://arxiv.org/abs/2202.11479v2).

To train an NMF model on the ESC50 dataset, use the `train_nmf.py` script. Run the command below:

```shell
python train_nmf.py hparams/nmf.yaml --data_folder /yourpath/ESC50 --save_period 30
```
Note that the variable `save_period` determines the period with which the reconstructions are saved for debugging purposes.

Additionally, we provide an L2I interpretation method for a convolutional classifier. To train this method on the ESC50 dataset, use the following command:

```shell
python train_l2i.py hparams/l2i_conv2d.yaml --data_folder /yourpath/ESC50
```
Note that the default l2i script uses the NMF dictionary specified in the hparams yaml file.

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

## Evaluation and Inference

### Out of distribution (OOD) tests

If you want to run tests on the OOD setting, you can use
```shell
python eval.py hparams/<config>.yaml --data_folder /yourpath/esc50 --overlap_type <mixtures/LJSpeech/white_noise> --add_wham_noise False --pretrained_interpreter yourpath/psi_model.ckpt
```

Note that overlap type should be either `mixture` (for contaminating signal to be set as other signals from ESC50), `LJSpeech` (for contaminating signal to be set as speech), or `white_noise` (for contaminating signal to be set as white noise). Please refer to the L-MAC paper for the performance obtained in each setting. Note that `yourpath/psi_model.ckpt` should point to the path of the model checkpoint you would like to use. The typical path for `yourpath/psi_model.ckpt` would be similar to `results/LMAC_cnn14/1234/save/CKPT+2024-06-20+16-05-44+00/psi_model.ckpt`.

Note also that `add_wham_noise` should be set to `False`.

Another thing to note is that if you use `--overlap_type LJSpeech`, you would need to specify the path via the variable `ljspeech_path`. If the LJSpeech dataset is not already downloaded on the path you specify, the code will automatically download it, and use the downloaded data.


### In distribution (ID) tests

If you want to run tests on the ID setting, you can use
```shell
python eval.py hparams/<config>.yaml --data_folder /yourpath/esc50 --add_wham_noise True --wham_folder /yourpath/wham_noise --pretrained_interpreter yourpath/psi_model.ckpt

```

This will evaluate the model using the test set contaminated with WHAM! noise samples.


### Single-sample inference

If you want to run inference on a single sample, you can use the following command:
```shell
python eval.py hparams/<config>.yaml --data_folder /yourpath/esc50 --add_wham_noise True --wham_folder /yourpath/wham_noise --pretrained_interpreter yourpath/psi_model.ckpt --single_sample $PATH_TO_WAV

```

---------------------------------------------------------------------------------------------------------

## Notes

- The recipe automatically downloads the ESC50 dataset. You only need to specify the path to which you would like to download it.

- All the necessary models are downloaded automatically for each training script.

---------------------------------------------------------------------------------------------------------

## Training Logs

| Method | Link |
| --- | --- |
| L-MAC | [Link](https://www.dropbox.com/scl/fo/k5r0zdrtkywamazrke2p1/AEP2D4Scu9mQ_McAxRYzWQQ?rlkey=qhwhe8729f2h2zbue88632f8n&st=vt316u20&dl=0)  |
| L-MAC FT| [Link](https://www.dropbox.com/scl/fo/kma3iznhjcyoco9slfwck/AMBmOXJAhiDUFs_dllXLaCQ?rlkey=drh04466lj1mca8qfrd31e14g&st=umd9ygj6&dl=0) |
| L2I CNN14 | [Link](https://www.dropbox.com/sh/cli2gm8nb4bthow/AAAKnzU0c80s_Rm7wx4i_Orza?dl=0) |
| L2I Conv2d | [Link](https://www.dropbox.com/sh/gcpk9jye9ka08n0/AAB-m10r1YEH0rJdUMrCwizUa?dl=0) |
| AMT-FocalNet |  [Link](https://www.dropbox.com/scl/fo/0hheboei1b35mlrhwj6mt/AOeCdNstN3h8UqFxv0abT7M?rlkey=kx0d1t5v5hqawqwr5ir9weihq&dl=0) |
| AMT-ViT |  [Link](https://www.dropbox.com/scl/fo/vlluiqiirlprl3oa7h4sj/APrEFgcIiWjdQhDUEZuNook?rlkey=bhswfspzklypu7k8ndh8lm3st&dl=0) |
| NMF Training | [Link](https://www.dropbox.com/sh/01exv8dt3k6l1kk/AADuKmikAPwMw5wlulojd5Ira?dl=0) |
| PIQ | [Link](https://www.dropbox.com/sh/v1x5ks9t67ftysp/AABo494rDElHTiTpKR_6PP_ua?dl=0) |
| PIQ-FocalNet | [Link](https://www.dropbox.com/scl/fo/6mvxb32f0g1i8b4lkdjoq/AGD1xNF8Of2_IXeEsbpXtQE?rlkey=llefue4rxalqyqwxqtwrn8qii&dl=0) |
| PIQ-ViT | [Link](https://www.dropbox.com/scl/fo/nz4lqwumgz03nanmf9xai/AI21fGwSOzsVvyegTJUEtz4?rlkey=40yjchqgkhcrhbxsa30m3rr6w&dl=0) |

## Citing

Please cite our [L-MAC paper](https://arxiv.org/abs/2403.13086) if you use it in your research:

```bibtex
@inproceedings{lmac,
  author={Francesco Paissan and Mirco Ravanelli and Cem Subakan},
  title={{Listenable Maps for Audio Classifiers}},
  year={2024},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
}
```


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


