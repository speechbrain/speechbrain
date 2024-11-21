# Data preparation of Libri-Light for SpeechBrain self-supervised-learning

This folder contains the script for preparing Libri-Light dataset for the SSL pre-training through SpeechBrain.

## How to run

1- Download the Libri-Light data (small, or medium, or  large) from

    https://github.com/facebookresearch/libri-light/tree/main/data_preparation


2- Use the cut_by_vad.py script from the Libri-Light repo to do the VAD of each downloaded split. For example, if you want to the use the small split

    python cut_by_vad.py --input_dir path_to_Libri-Light_small --output_dir Libri-Light_small_vad --target_len_sec 20

   If you want your audio clips in the SSL pre-training to have a maximum length of 20 seconds, then it is recommended to use "--target_len_sec" equal or less than 20 seconds

3- Use the vad output_dir in step 2 as the input_dir to call make_librilight_csv.py to generate the csv file. For example

    python make_librilight_csv.py --input_dir Libri-Light_small_vad --output_dir results --max_length 20 --n_processes 128


4- Now, you can use the generated train.csv in the output_dir in step 3 as the "train_csv" in any SpeechBrain SSL pre-training yaml file