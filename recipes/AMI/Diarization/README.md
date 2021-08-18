# Speaker Diarization on AMI corpus
This directory contains the scripts for speaker diarization on the AMI corpus (http://groups.inf.ed.ac.uk/ami/corpus/).

## Extra requirements
The code requires sklearn as an additional dependency. 
To install it, type: `pip install sklearn`

## How to run
Use the following command to run diarization on AMI corpus.
`python experiment.py hparams/ecapa_tdnn.yaml` or `python experiment.py hparams/xvectors.yaml` depending upon the model used.


## Speaker Diarization using Deep Embedding and Spectral Clustering
The script assumes the pre-trained model. Please refer to speechbrain/recipes/VoxCeleb/SpeakerRec/README.md to know more about the available pre-trained models that can easily be downloaded. You can also train the speaker embedding model from scratch using instructions in the same file. 


## Best performance in terms of Diarization Error Rate (DER).
The forgiveness collar of 0.25 sec is used and overlaps are ignored while evaluation.

| System | Mic. | Orcl. (Dev) | Orcl. (Eval) | Est. (Dev) | Est. (Eval)
|----------- | ------------ | ------ |------| ------| ------ |
| ECAPA-TDNN + SC | HeadsetMix | 2.02% | 1.78% | 2.43% | 4.03% |
| ECAPA-TDNN + SC | LapelMix | 2.17% | 2.36% | 2.34% | 2.57% |
| ECAPA-TDNN + SC | Array-1 | 2.95% | 2.75% | 3.07% | 3.30% |

For the complete set of analyses, please refer to our paper given below.

## Citation

Paper Link: [ECAPA-TDNN Embeddings for Speaker Diarization](https://arxiv.org/pdf/2104.01466.pdf)

If you find the code useful in your work, please cite:

    @misc{dawalatabad2021ecapatdnn,
          title={ECAPA-TDNN Embeddings for Speaker Diarization},
          author={Nauman Dawalatabad and Mirco Ravanelli and Francois Grondin and Jenthe Thienpondt and Brecht Desplanques and Hwidong Na},
            year={2021},
          eprint={2104.01466},
          archivePrefix={arXiv},
          primaryClass={eess.AS},
          note={arXiv:2104.01466}
    }
