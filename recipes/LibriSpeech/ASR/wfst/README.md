# WFST Decoding Based on k2
This repository aims to add WFST decoding based on k2 for speechbrain. You can know more details about how k2 implements WFST from [k2](https://github.com/k2-fsa/k2) and [icefall](https://github.com/k2-fsa/icefall).

## Environment
Suggest run this script with a conda environment (Linux).  You can also config your environment from this [url](https://k2-fsa.github.io/k2/) based on your reality.

1. conda create -n k2-python python=3.8
2. source activate k2-python
3. pip install torch==1.8.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
4. conda install -c k2-fsa -c pytorch -c conda-forge k2 python=3.8 cudatoolkit=10.2 pytorch=1.8.1
5. pip install speechbrain

If there are some other packages missing, you can install them as follows:

1. pip install xxx or conda install xxx


## Dataset

