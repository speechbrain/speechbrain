# Speaker Diarization on AMI corpus
This directory contains the scripts for speaker diarization on the AMI corpus (http://groups.inf.ed.ac.uk/ami/corpus/).

# Extra requirements
The code requires sklearn as an additional dependency. To install it, type:
pip install sklearn

# How to run
python experiment.py hparams/ecapa_tdnn.yaml

# Speaker Diarization using Deep Embedding and Spectral Clustering
The script assumes the pre-trained model. Please refer to speechbrain/recipes/VoxCeleb/SpeakerRec/README.md to know more about the available pre-trained models that can easily be downloaded.
You can also train the speaker embedding model from scratch using instructions in the same file. Use the following command to run diarization on AMI corpus.

`python experiment.py hparams/xvectors.yaml`
`python experiment.py hparams/ecapa_tdnn.yaml`

# Performance Summary using Xvector model trained on VoxCeleb1+VoxCeleb2 dataset
Xvectors : Dev = 4.34 % | Eval = 4.45 %
ECAPA   :  Dev = 2.19 % | Eval = 2.74 %
ECAPA_big: Dev = 2.16 % | Eval = 2.72 %

