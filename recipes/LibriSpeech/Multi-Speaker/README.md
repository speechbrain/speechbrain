# Overlapping LibriSpeech Data Creation

Here, we combine a variable number of 5s unique speaker LibriSpeech samples and create overlapping speech samples to be used in speaker count applications. 

## Usage

To run, use the following command:

```bash
python data_creation.py hparams/data_params.yaml --original_data_folder <link-to-folder> --new_data_folder <link-to-other-folder>
```