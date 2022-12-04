# **Microsoft DNS dataset**
- **STEP 1**: Download DNS dataset.
- **STEP 2**: Synthesize noisy data.
- **STEP 3**: Begin training.

Source: [Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022](https://github.com/microsoft/DNS-Challenge)

This script downloads the Real-time DNS track data and de-compresses it. The compressed data takes around 550 GB of disk space and when de-compressed you would need 1 TB to store audio files.

The de-compressed directory structure and sizes of datasets are:
```
datasets_fullband 892G
+-- dev_testset 1.7G
+-- impulse_responses 5.9G
+-- noise_fullband 58G
\-- clean_fullband 827G
    +-- emotional_speech 2.4G
    +-- french_speech 62G
    +-- german_speech 319G
    +-- italian_speech 42G
    +-- read_speech 299G
    +-- russian_speech 12G
    +-- spanish_speech 65G
    +-- vctk_wav48_silence_trimmed 27G
    \-- VocalSet_48kHz_mono 974M
```

However this is not the end, the downloaded clean-audio files, RIRs, and noisy-audio files are further used to synthesize clean-noisy audio pairs for training. See `noisyspeech_synthesizer` on how to synthesize noisy files from clean audio and noise audio files.

**NOTE**
- This dataset download process can be extremely time-consuming. With a total of 127 splits (bz2 files), the script downloads each split in a serial order. This calls for the need to implement a parallel data-downloading method. This is helpful especially when you have access to a large cluster. You can download all 127 splits and decompress them at once by using array job submission.

## **Downloading Real-time DNS track dataset**
The DNS dataset can be downloaded by running the script below
```
python dns_download.py
```
The compressed files are downloaded in `DNS-compressed` and further decompressed audio files can be found in `DNS-dataset`.

### TODOs
0. Downsample RIRs to 16K
1. Implement resumable downloads
2. Parallel downloading
3. Dynamic Mixing
4. DNSMOS evaluation of
    - DEV noisy files
    - enhanced files using baseline NSNET-2 model
    - enhanced files using SepFormer
5. Final results need to be added.
