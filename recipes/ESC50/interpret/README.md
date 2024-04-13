# Postdoc interpretability - ESC50 Dataset

The objective of postdoc interpretability is to offer an explanation regarding the decision made by a pre-trained classifier.

The interpreter is a neural network that generates an additional signal in its output, aiming to assist users in better comprehending why a specific prediction was made. You can find some examples [here](https://piqinter.github.io/).

![image](https://github.com/ycemsubakan/speechbrain-1/assets/16886998/8199f0fb-66ee-4f5a-87ee-349695f7e982)


The recipes implements the posthoc interpretability techniques mentioned below. They utilize pre-trained models obtained from `recipes/ESC50/classification/`, which are automatically downloaded from [HuggingFace](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech). If necessary, you have the option to train your own classifier by following the instructions provided in the reference README.

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

## Supported Methods

### Posthoc Interpretability via Quantization (PIQ)

PIQ utilizes vector quantization on the classifier's representations to reconstruct predictions. For more details, refer to the [PIQ paper](https://arxiv.org/abs/2303.12659). You can visit the companion website for PIQ [here](https://piqinter.github.io/). To train PIQ on a convolutional classifier using the ESC50 dataset, use the `train_piq.py` script. Run the following command:

```python
python train_piq.py hparams/piq.yaml --data_folder=/yourpath/ESC50
```

Check out an example training run [here](https://www.dropbox.com/sh/v1x5ks9t67ftysp/AABo494rDElHTiTpKR_6PP_ua?dl=0). You can also find samples on the [companion website](https://piqinter.github.io/).

### Listen to Interpret (L2I)

L2I employs Non-Negative Matrix Factorization to reconstruct the classifier's hidden representation and generate an interpretation audio signal for the classifier decision. Read more about L2I in the [L2I paper](https://arxiv.org/abs/2202.11479v2). To train an NMF model on the ESC50 dataset, use the `train_l2i.py` script. Run the command below:

```python
python train_nmf.py hparams/nmf.yaml --data_folder=/yourpath/ESC50
```

You can find an example training run [here](https://www.dropbox.com/sh/01exv8dt3k6l1kk/AADuKmikAPwMw5wlulojd5Ira?dl=0).

Additionally, we provide an L2I interpretation method for a convolutional classifier. To train this method on the ESC50 dataset, use the following command:

```python
python train_l2i.py hparams/l2i_conv2dclassifier.yaml --data_folder=/yourpath/ESC50
```

An example training run is available [here](https://www.dropbox.com/sh/gcpk9jye9ka08n0/AAB-m10r1YEH0rJdUMrCwizUa?dl=0).

Lastly, we offer the training script for the L2I interpretation method on CNN14. To run this, execute the following command:

```shell
python train_l2i.py hparams/l2i_cnn14.yaml --data_folder /yourpath/ESC50
```

You can find an example training run in the [provided link](https://www.dropbox.com/sh/cli2gm8nb4bthow/AAAKnzU0c80s_Rm7wx4i_Orza?dl=0).

### Notes

- The recipe automatically downloads the ESC50 dataset, so you only need to specify the download path.
- All the necessary models are downloaded automatically for each training script.

# How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:
```shell
python train.py hparams/{hparam_file}.py --data_folder /yourpath/ESC50 --test_only
```

# Inference Interface (on HuggingFace)
You can access the inference interface for the PIQ method [here](https://huggingface.co/speechbrain/PIQ-ESC50/).

You can notice that the interpreter requires an input signal (such as a complex audio recording containing multiple mixed sounds), and the output is another audio signal that aims to provide an explanation for the classifier's decision.


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing PIQ**
Please cite our paper on PIQ if you use it in your research.

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

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

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

