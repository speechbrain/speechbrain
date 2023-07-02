Dear User,

We are thrilled to introduce our latest release - the **RescueSpeech** audio dataset, comprising authentic German speech recordings obtained from simulated search and rescue (SAR) exercises. The dataset contains manually annotated recordings from native German speakers, which were initially captured at 44.1 kHz and later down-sampled to 16 kHz to obtain a set of mono-speaker-single channel audio recordings. In order to protect the identity of the speakers, their names have been anonymized.

The RescueSpeech dataset is divided into two sets, each designed for different tasks: Automatic Speech Recognition (ASR) and Speech Enhancement.

1. `Task_ASR.tar.gz`: For the ASR task, the dataset spans a duration of 1 hour and 36 minutes. It comprises a collection of clean-noisy pairs, where the noisy utterances are created by introducing contaminations from five different noise types sourced from the AudioSet dataset. These noise types include emergency vehicle siren, breathing, engine, chopper, and static radio noise. To match the 2412 clean utterances in the dataset, we have synthesized an equal number of corresponding noisy utterances. Additionally, we have provided the noise waveform files used to create the noisy utterances, ensuring transparency and reproducibility in the research community.

2. `Task_enhancement.tar.gz`: The Speech Enhancement task dataset is larger in size compared to the ASR dataset. The primary objective of this dataset is to facilitate the fine-tuning of speech enhancement models, particularly for the five SAR noise types mentioned earlier: emergency vehicle siren, breathing, engine, chopper, and static radio noise. Given the limited duration of clean audio available (1 hour and 36 minutes), we have synthesized multiple noisy utterances with varying noise types and signal-to-noise ratio (SNR) levels, all derived from a single clean utterance. This augmentation approach allows us to generate a more extensive dataset for speech enhancement purposes while preserving the original speaker distribution.

By providing these diverse datasets, we aim to support advancements in ASR and Speech Enhancement research, enabling the development and evaluation of robust systems that can handle real-world scenarios encountered during search and rescue operations.

## Main contact person
------------------
For any inquiries related to the dataset, please reach out to
Bernd Kiefer: bernd.kiefer@dfki.de

Other contact people
--------------------
- Ivana Kruijf‑Korbayová: ivana.kruijff@rettungsrobotik.de
- Sangeet Sagar: sangeetsagar2020@gmail.com

## Task: ASR- Dataset details
---------------
- Total number of recordings: 2412
- Total duration: 1:36:10
- Number of speakers: 26
- Number of recordings where speaker is undetermined (indicated with ?): 38
- Average length of dataset: 2.39 sec
- Longest duration: 15 sec
- Shortest duration: 0.28 sec

To obtain a train/test/dev set, we perform a stratified sampling technique that ensures that the valid/test set contains a representative sample of speakers from the overall population. We first identify a set of unique speakers in the dataset and then randomly sample a subset of those speakers to be included in the test/dev set. The remaining speakers are assigned to the train set.

Train Split
-----------
- Total number of files: 1591
- Total duration: 61.86 mins
- Total number of speakers: 17
- Speakers involved: spk01, spk02, spk05, spk07, spk08, spk09, spk10, spk11, spk12, spk13, spk16, spk19, spk20, spk21, spk22, spk23, spk25

Test Split
-----------
- Total number of files: 576
- Total duration: 24.68 mins
- Total number of speakers: 5
- Speakers involved- spk03, spk06, spk15, spk24, ?

Dev Split
-----------
- Total number of files: 245
- Total duration: 9.61 mins
- Total number of speakers: 4
- Speakers involved- spk04, spk14, spk17, spk18


This table represents the number of recordings in each of the three sets (train, test, and dev) for each speaker ID. The speaker IDs are listed in the first column, while the number of recordings for each speaker in each set is listed in the corresponding column.


| Speaker ID | train.tsv | test.tsv | dev.tsv | **Total** |
|:-----------|:----------|:---------|:--------|:------|
| ?          | 0         | 38       | 0       | 38    |
| spk01      | 211       | 0        | 0       | 211   |
| spk02      | 502       | 0        | 0       | 502   |
| spk03      | 0         | 344      | 0       | 344   |
| spk04      | 0         | 0        | 204     | 204   |
| spk05      | 266       | 0        | 0       | 266   |
| spk06      | 0         | 164      | 0       | 164   |
| spk07      | 257       | 0        | 0       | 257   |
| spk08      | 25        | 0        | 0       | 25    |
| spk09      | 48        | 0        | 0       | 48    |
| spk10      | 24        | 0        | 0       | 24    |
| spk11      | 27        | 0        | 0       | 27    |
| spk12      | 7         | 0        | 0       | 7     |
| spk13      | 7         | 0        | 0       | 7     |
| spk14      | 0         | 0        | 12      | 12    |
| spk15      | 0         | 15       | 0       | 15    |
| spk16      | 8         | 0        | 0       | 8     |
| spk17      | 0         | 0        | 4       | 4     |
| spk18      | 0         | 0        | 25      | 25    |
| spk19      | 7         | 0        | 0       | 7     |
| spk20      | 37        | 0        | 0       | 37    |
| spk21      | 102       | 0        | 0       | 102   |
| spk22      | 13        | 0        | 0       | 13    |
| spk23      | 49        | 0        | 0       | 49    |
| spk24      | 0         | 15       | 0       | 15    |
| spk25      | 1         | 0        | 0       | 1     |
| **Totals**     | 1591      | 576      | 245     | 2167  |

** ? indicates undetermined speakers.

## Task: Speech enhancement- Dataset details
---------------
- Noises used:
    - Static and radio noise
    - Emergency vehicle and siren noise
    - Engine
    - Chopper
    - Breathing

| Set       |   # wav files     |   Length      |
|:----------|:------------------|:--------------|
|Train      |   4501            |   7.2 HRS     |
|Valid      |   1351            |   130 mins    |
|Test       |   1351            |   130 mins    |


Thank You


## Acknowledgment
---------------
This work was supported under the project A-DRZ: Setting up the German Rescue Robotics Center and funded by the German Ministry of Education and Research (BMBF), grant No. I3N14856.
