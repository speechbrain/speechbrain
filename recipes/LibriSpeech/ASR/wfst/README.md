# WFST Decoding Based on k2
This repository aims to add WFST decoding based on k2 for speechbrain with python.  You can know more details about how k2 implements WFST from [k2](https://github.com/k2-fsa/k2) and [icefall](https://github.com/k2-fsa/icefall). Here, we use the transformer trained by speechbrain as our acoustic model to compute the output probabilities. Based on the output probabilities, we implement WFST decoding with k2. 

## Environment
Suggest run this script with a conda environment (Linux).  You can also config your environment according to this [url](https://k2-fsa.github.io/k2/) based on your reality.
```
1. conda create -n k2-python python=3.8
2. source activate k2-python
3. pip install torch==1.8.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
4. conda install -c k2-fsa -c pytorch -c conda-forge k2 python=3.8 cudatoolkit=10.2 pytorch=1.8.1
5. pip install speechbrain
```
If there are some other packages missing, you can install them as follows:
```
pip install xxx or conda install xxx
```

## Dataset
You can change the data_dir to your own librispeech data directory in test_ctc.py and test_hlg.py .

## Testing
You just need run the following command:
```
bash run.sh
```
The run.sh can unify all python files and complete them with just one command.

## Results
CUDA_VISIBLE_DEVICES='0' python3 test_ctc.py

||test-clean|test-other|
|--|--|--|
|WER| 2.57% | 5.94% |

CUDA_VISIBLE_DEVICES='0' python3 test_hlg.py
|WER|
||test-clean|test-other|
|--|--|--|
|WER| 2.57% | 5.94% |

