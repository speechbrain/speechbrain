# **Speech enhancement with Microsoft DNS dataset**
This folder contains the recipe for speech enhancement on Deep Noise Suppression (DNS) Challenge 4 (ICASSP 2022) dataset using SepFormer.

For data download and prepration, please refer to the `README.md` in `recipes/DNS/`

## **Start training**
```
python train.py hparams/sepformer-dns-16k.yaml --data_folder <path/to/synthesized_shards_data> --baseline_noisy_shards_folder <path/to/baseline_dev_shards_data>
```
## **DNSMOS Evaluation on baseline-testclips**
*Reference: [Offical repo](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) <br>*
Download the evalution models from [Offical repo](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) and save it under `DNSMOS`. Then, to run DNSMOS evalution on the baseline-testclips saved in the above step.
```
# Model=SepFormer
python dnsmos_local.py -t results/sepformer-enhancement-16k/1234/save/baseline_audio_results/enhanced_testclips/ -o dnsmos_enhance.csv

# Model=Noisy
python dnsmos_local.py -t <path-to/datasets_fullband/dev_testset/noisy_testclips/> -o dnsmos_noisy.csv
```

## **Results**
1. The DNS challenge doesn't provide the ground-truth clean files for dev test. Therefore, we randomly separate out 5% of training set as valid set so that we can compute valid stats like Si-SNR and PESQ during validation. Here we show validation performance.

      | Sampling rate | Valid Si-SNR | Valid PESQ | HuggingFace link	| Full Model link |
      |---------------|--------------|------------|-------------------|------------|
      | 16k           | -10.6        | 2.06       | [HuggingFace](https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement) |  https://www.dropbox.com/sh/d3rp5d3gjysvy7c/AACmwcEkm_IFvaW1lt2GdtQka?dl=0          |

2. Evaluation on DNS4 2022 baseline dev set using DNSMOS.

    | Model      | SIG    | BAK    | OVRL   |
    |------------|--------|--------|--------|
    | Noisy      | 2.984  | 2.560  | 2.205  |
    | Baseline: NSNet2| 3.014  | 3.942  | 2.712  |
    | **SepFormer**  | 2.999  | 3.076  | 2.437  |

We performed 45 epochs of training for the enhancement using an 8 X RTXA6000 48GB GPU. On average, each epoch took approximately 9.25 hours to complete. **Consider training it for atleast 90-100 epochs for superior performance.**

**NOTE**
- Refer [NSNet2](https://github.com/microsoft/DNS-Challenge/tree/5582dcf5ba43155621de72a035eb54a7d233af14/NSNet2-baseline) on how to perform enhancement on baseline dev set (noisy testclips) using the baseline model- NSNet2.

## **Computing power**
Kindly be aware that in terms of computational power, training can be extremely resource demanding due to the dataset's large size and the complexity of the SepFormer model. To handle the size of 1300 hours of clean-noisy pairs, we employed a multi-GPU distributed data-parallel (DDP) training scheme on an Nvidia 8 X RTXA6000 48GB GPU. The training process lasted for 17 days, for just 45 epochs.

## **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


## **Citing SpeechBrain**
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


**Citing SepFormer**
```bibtex
@inproceedings{subakan2021attention,
      title={Attention is All You Need in Speech Separation},
      author={Cem Subakan and Mirco Ravanelli and Samuele Cornell and Mirko Bronzi and Jianyuan Zhong},
      year={2021},
      booktitle={ICASSP 2021}
}
```
