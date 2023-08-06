# **Microsoft DNS-4 dataset**
- STEP 1: Download DNS dataset.
- STEP 2: Synthesize noisy data.
- STEP 3: Begin training.

Source-- [Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022](https://github.com/microsoft/DNS-Challenge)

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

## Required disk space
The `dns_download.py` download script downloads the Real-time DNS track data and de-compresses it. The compressed data takes around 550 GB of disk space and when de-compressed you would need 1 TB to store audio files.
However this is not the end, the downloaded clean-audio files, RIRs, and noisy-audio files are further used to synthesize clean-noisy audio pairs for training. This means further space will be needed to store the synthesized clean-noisy-noise audio.

**NOTE**
- This dataset download process can be extremely time-consuming. With a total of 126 splits (train, noise and dev data), the script downloads each split in a serial order. The script also allows concurrent data download (by enabling `--parallel_download` param) by using multiple threads (equal to number of your CPU cores). This is helpful especially when you have access to a large cluster. (Alternatively, you can download all 126 splits and decompress them at once by using array job submission.)

## Step 1: **Downloading Real-time DNS track dataset**
The DNS dataset can be downloaded by running the script below
```
python dns_download.py --compressed_path DNS-dataset --decompressed_path DNS-compressed
```
To use parallel downloading
```
python dns_download.py --compressed_path DNS-dataset --decompressed_path DNS-compressed --parallel_download
```

The compressed files are downloaded in `DNS-compressed` and further decompressed audio files can be found in `DNS-dataset`.

## Step 2: **Synthesize noisy data**
See `noisyspeech_synthesizer` on how to synthesize noisy files from clean audio and noise audio files.

### TODOs
1. Dynamic Mixing
2. DNSMOS evaluation of
    - DEV noisy files
    - enhanced files using baseline NSNET-2 model
    - enhanced files using SepFormer
3. Final results need to be added.
