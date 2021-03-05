# TIMIT ASR with Transducer models.
This folder contains the scripts to train an RNNT system using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1


# Extra-Dependencies
Before running this recipe, make sure numba is installed. Otherwise, run: 
pip install numba

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Val. PER | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 2021-02-06 | train.yaml |  13.11 | 15.08 | https://drive.google.com/drive/folders/1g3T6zK2o9XTEa_GTw0aoAkRqhg1_BVQ3?usp=sharing | 1xV100 16GB |

# Training Time
About 3 min and 20 sec for each epoch with a  TESLA V100.

