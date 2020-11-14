# Speaker Diarization on AMI corpus
This directory contains the scripts for speaker diarization on AMI corpus (http://groups.inf.ed.ac.uk/ami/corpus/).


# Speaker Diarization using Deep Embedding and Spectral Clustering
The scripts assumes the pre-trained model. Please refer to speechbrain/recipes/VoxCeleb/SpeakerRec/README.md to know more about the available pre-trained models that can easily be downloaded.
You can also train the speaker embedding model from scratch using instructions in the same file. Use the following command to run diarization on AMI corpus.

`python speaker_diary.py hyperparams/diarization.yaml`

# Performance Summary using Xvector model trainined on VoxCeleb1+VoxCeleb2 dataset
[Diarization Error Rate on AMI. Condition Orcale VAD and Oracle number of speakers]
| System          | AMI-Dev    | AMI-Eval |
|-----------------|------------|------|
| Xvector + SC  | 4.45 % | 4.53% |
