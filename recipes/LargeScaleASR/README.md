# The Loquacious Set: 25,000 hours of transcribed and heterogeneous English speech recognition data for research and commercial use.

This folder provides a recipe for building The Loquacious Set (TLS) as well as
performing speech recognition with a conformer encoder-decoder architecture.

TLS is made of 6 subsets:
1. **large** contains 25,000 hours of read / spontaneous and clean / noisy transcribed speech.
2. **medium** contains 2,500 hours of read / spontaneous and clean / noisy transcribed speech.
3. **small** contains 250 hours of read / spontaneous and clean / noisy transcribed speech.
4. **clean** contains 13,000 hours of read and clean / less noisy transcribed speech.
5. **dev** contains 15 hours (more details in the next section).
6. **test** contains 21 hours (more details in the next section).

The Loquacious Set can be loaded following three different solutions:
1. SpeechBrain CSV.
2. HuggingFace CSV datasets.
3. HuggingFace shards (**we may end up with only this one**).

The total dataset 3.x TB. The double is needed in case of extraction (like HuggingFace).

## Data description

TLS is a mix of 5 existing dataset with permissive licences. The way it is mixed
is described in the following table:

| Dataset       | Amount Taken (large/medium/small/dev/test) | License |
| ------------- | ------------- | ------------- |
| VoxPopuli | 550/500/50/5/7 | CC0  |
| LibriHeavy | 11,000/500/50/0/0 | CC BY 4.0 |
| Librispeech (dev-/test-other) | 0/0/0/5/7 | CC BY 4.0 |
| yodas | 6,100/500/50/0/0 |  CC BY 3.0 |
| people's speech | 5,900/500/50/0/0 | CC-BY 4.0 |
| CommonVoice 18.0 | 1660/500/50/5/7 | CC0 |

*For dev and tests splits, only data from the corresponding dev and test sets of the considered dataset is used (i.e. not extracted from the train except for YODAS). For YODAS we extract data from the en003 split and verify the audio/transcription manually to form the dev/test partitions*

More information relative to each dataset is given as:

- [**voxpopuli**](https://arxiv.org/abs/2101.00390): we follow the standard SpeechBrain data preparation.
- [**LibriHeavy**](https://arxiv.org/html/2309.08105v2): samples are randomly selected, but we follow the standard data preparation.
- [**Librispeech**](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf): Librispeech is only used for the validation and test sets of The Loquacious set. More precisely, we extract samples from *dev-others* and *test-others* as they are the most challenging subsets.
- [**YODAS**](https://arxiv.org/abs/2406.00899): The YODAS dataset is unfortunately unreliable. Indeed, audio are crawled from YouTube, and a lot of them (almost half) do not have the correct language. We used a [SpeechBrain language ID model](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) to make sure that we only integrate samples where people speak in English. Transcriptions have also been heavily normalised (see next section). We decided arbitrarily to use the *en000* and *en001* subsets of Yodas. Transcriptions may be a bit noisy. This is why this dataset is excluded from the dev and test sets of The Loquacious Set.
- [**People's Speech**](https://huggingface.co/datasets/MLCommons/peoples_speech): Only the *clean* subset of this dataset is used in The Loquacious Set as the transcriptions there already have errors. This is why this dataset is excluded from the dev and test sets of The Loquacious Set.
- [**CommonVoice 18.0**](https://commonvoice.mozilla.org/en): We removed a few speakers that had too many samples (above 9000 samples) to avoid any bias. Aside from this, we used only samples coming from the *validated* csv to ensure an optimal level of transcriptions. Text was also heavily normalised (see next section).

### Text and audio normalisation

Some of the above datasets, in particular People's Speech, Yodas and CommonVoice have very little normalisation. This is an important issue as the pronunciation is then either incorrect or uncertain. We normalised all the sentences to ensure a set of characters containing only the standard 26 letter of the European alphabet plus the "'". Numerical values were converted to text using the [Nemo text processing WFST tool](https://github.com/NVIDIA/NeMo-text-processing). The rest of the text was properly filtered to remove symbols, youtube annotations like "applause" or many others elements. When sentences were too noisy, we simply decided to remove them (e.g. too many symbols). The text normalisation can be found in *speechbrain.utils.text_normalisation*.

Audio are all .wav files or .flac files with a sample rate of 16kHz. We chunked and created smaller audio files from long ones based on start and stop supervision from the different manifests of the datasets (this is necessary for HuggingFace). Language ID with a [SpeechBrain language ID model](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) was performed on Yodas.