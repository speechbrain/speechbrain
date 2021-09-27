# WFST Decoding Based on k2
This repository aims to add WFST decoding based on k2 for speechbrain.

## Installation
Suggest run this script with a conda environment. (Linux) 

1. conda create -n k2-python python=3.8
2. source activate k2-python
3. pip install torch==1.8.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
4. conda install -c k2-fsa -c pytorch -c conda-forge k2 python=3.8 cudatoolkit=10.2 pytorch=1.8.1
5. pip install speechbrain

if there are some other packages missing, you can install them as follows:

1. pip install xxx

## 

