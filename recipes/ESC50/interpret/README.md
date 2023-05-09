# Summary

This recipe implements the following posthoc interpretability techniques. This trainings run on pre-trained models from `recipes/ESC50/classification/` automatically downloaded from HuggingFace Hub. If needed, you can train your own classifier with the instruction in the reference README.

- [Posthoc Interpretability via Quantization]() which makes use of vector quantization on the classifier's representations to reconstruct the predictions. The companion website for PIQ is [here](https://piqinter.github.io/). The corresponding training script is `train_piq.py`.
    * *The training script for PIQ:* This script trains PIQ on a convolutional classifier working on the ESC50 dataset. To run this, you can use the command `python train_piq.py hparams/piq.yaml --data_folder /yourpath/ESC50`. An example training run can be found in [`https://drive.google.com/drive/folders/10u78s0tUHT5GfuqmcRdqzdEPx6D965kt?usp=share_link`](https://drive.google.com/drive/folders/10u78s0tUHT5GfuqmcRdqzdEPx6D965kt?usp=share_link). Companion website with samples can be found [here](https://piqinter.github.io/).
- [Listen to Interpret](https://arxiv.org/abs/2202.11479v2) which makes use of Non-Negative Matrix Factorization to reconstruct the classifier hidden representation in order to provide an interpretation audio signal for the classifier decision. The corresponding training script is `train_l2i.py`.

	* *The training script NMF on ESC50:* This script trains an NMF model on the ESC50 dataset. To run this, you can use the command `python train_nmf.py hparams/nmf.yaml --data_folder /yourpath/ESC50`. An example training run can be found in `[https://drive.google.com/drive/folders/1cUC5vpZVMuZi6bGhLHduoi6I-tuh4Bwu?usp=share_link`](https://drive.google.com/drive/folders/1cUC5vpZVMuZi6bGhLHduoi6I-tuh4Bwu?usp=share_link).

    * *The training script for L2I interpretation method on convolutional classifier:*: This script trains the L2I method on the ESC50 dataset, interpreting a convolutional model. To run this you can use the command `python train_l2i.py hparams/l2i_conv2dclassifier.yaml --data_folder /yourpath/ESC50`. An example training run can be found in [`https://drive.google.com/drive/folders/1059ghU9MZOUx9cZ5velwkefD8MDsO1AK?usp=share_link`](https://drive.google.com/drive/folders/1059ghU9MZOUx9cZ5velwkefD8MDsO1AK?usp=share_link).

	* *The training script for L2I interpretation method on CNN14:*: This script trains the L2I method on the ESC50 dataset, interpreting a CNN14 model. To run this you can use the command `python train_l2i.py hparams/l2i_cnn14.yaml --data_folder /yourpath/ESC50`. An example training run can be found in [`https://drive.google.com/drive/folders/1059ghU9MZOUx9cZ5velwkefD8MDsO1AK?usp=share_link`](https://drive.google.com/drive/folders/1059ghU9MZOUx9cZ5velwkefD8MDsO1AK?usp=share_link).

Note that:
    - the recipe automatically downloads the ESC50 dataset. You only need to specify the path to which you would like to download it;
    - all of the necessary models are downloaded automatically for each training script.



# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


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
