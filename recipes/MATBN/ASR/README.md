# MATBN ASR with Transformers.
This folder contains recipes for tokenization and speech recognition with MATBN.

### How to run
1. Train a tokenizer. The tokenizer takes in input the training transcripts and determines the subword units that will be used for both acoustic and language model training.

    ```
    cd ../Tokenizer
    python train.py hparams/tokenizer_char5k.yaml --dataset_folder=<your data location>
    ```
2. Train a language model. (select one of RNNLM and TransformerLM)

    ```
    cd ../LM
    python train.py hparams/RNNLM.yaml --tokenizer_file=../Tokenizer/results/tokenizer_char5k/5000_char.model  --data_folder=../Tokenizer/results/prepare
    ```
    or
    ```
    cd ../LM
    python train.py hparams/TransformerLM.yaml --tokenizer_file=../Tokenizer/results/tokenizer_char5k/5000_char.model --data_folder=../Tokenizer/results/prepare
    ```

3. Train the speech recognizer. (select one of RNNLM and TransformerLM)

    ```
    python train.py hparams/transformer_RNNLM.yaml --data_folder=../Tokenizer/results/prepare --tokenizer_file=../Tokenizer/results/tokenizer_char5k/5000_char.model --lm_file=<lm ckpt location>
    ```
    or
    ```
    python train.py hparams/transformer_TransformerLM.yaml --data_folder=../Tokenizer/results/prepare --tokenizer_file=../Tokenizer/results/tokenizer_char5k/5000_char.model --lm_file=<lm ckpt location>
    ```

# Performance summary
Results are reported in terms of Character Error Rate (CER).

| hyperparams file | Test CER | GPUs |
|:--------------------------:| :-----:| :-----: |
| transformer_RNNLM.yaml | 8.41 | 1xGTX1080 8GB |
| transformer_TransformerLM.yaml | 8.25 | 1xGTX1080 8GB |

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