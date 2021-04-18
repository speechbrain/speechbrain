# SpeechEnhancement with the DNS dataset (spectral map).
This folder contains the scripts to train a speech enhancement system with spectral map.
You can download the dataset from here: https://github.com/microsoft/DNS-Challenge

# How to run
python train.py train/params_CNNTransformer.yaml  
python train.py train/params_CNN.yaml 

# Results
| Release | hyperparams file | STOI | PESQ | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 |  params_CNN.yaml |  --.- | --.- | Not Available | 1xV100 32GB |
| 20-05-22 |  params_CNNTransformer.yaml |  --.- | --.- | Not Available | 1xV100 32GB |

# Training Time
About -- for each epoch with a TESLA V100.
