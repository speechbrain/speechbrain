# LibriSpeech ASR with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the LibriSpeech dataset;
Before running this recipe, make sure numba is installed (pip install numba)
You can download LibriSpeech at http://www.openslr.org/12

# How to run it
python train.py train/train.yaml

# Librispeech 100H Results

| Release | hyperparams file | Val. CER | Val. WER | Test WER (test clean) | Model link | GPUs |
|:-------------:|:---------------------------:| ------:| :-----------:| :------------------:| --------:| :-----------:|
| 2020-10-22 | train.yaml |  5.2 | GS: 11.45 | BS (beam=4): 11.03 | Not Available | 1xRTX-8000 48GB |

# Training time
About N for each epoch with a  TESLA V100.
