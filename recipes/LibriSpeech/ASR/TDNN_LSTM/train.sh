#!/bin/bash

# First prepare the G language models (3-gram for HLG and 4-gram for rescoring).


seed=1112
device="cuda:1,2,3"

# TODO: Maybe download librispeech and musan here (similar to icefall's prep)
data_folder="<path-to-LibriSpeech>"
musan_folder="<path-to-musan>"

# if [[ ! -f $data_folder ]] ; then
if [[ ! -f $data_folder || ! -f $musan_folder ]] ; then
  echo "$0: make sure to set data_folder and musan_folder inside this script."
  echo "$0: (musan is not really required so you could just comment this check if you don't want to use it)."
  exit 1;
fi

# train_sets="train-clean-100"  # train-clean-360 train-other-500"
train_sets="train-clean-100 train-clean-360 train-other-500"
# Construct a train_splits variable that will be the same as train_sets but
# will be in a format recognized by python's argparse
train_splits="[\"$(echo $train_sets | sed 's/ /","/g')\"]"

output_folder=results/train_k2_decoder_adamw/$seed
save_folder=$output_folder/save/

token_type=unigram
n_bpe_units=500
bpe_dir=$save_folder/lang_bpe_$n_bpe_units/

dl_dir=$output_folder/

echo "Preparing the data for the language models. $dl_dir"

mkdir -p $dl_dir/lm
mkdir -p $bpe_dir

# Now use the lexicon.py file to create the words.txt file
# This is needed for the creation of the G and G_4_gram
python lexicon.py \
    --bpe-model-path=$bpe_dir/"$n_bpe_units"-$token_type.model \
    --librispeech-dir=$data_folder \
    --train-sets=$train_splits \
    --build-words || exit 1;

word_txt_path=$bpe_dir/words.txt

# lm_files="3-gram.pruned.1e-7.arpa.gz 3-gram.arpa.gz 4-gram.arpa.gz librispeech-lm-norm.txt.gz librispeech-lexicon.txt librispeech-vocab.txt"
lm_files="3-gram.pruned.1e-7.arpa.gz 3-gram.arpa.gz 4-gram.arpa.gz librispeech-lm-norm.txt.gz"

for f in $lm_files; do
    if [ ! -f $dl_dir/lm/$f ] ; then
        echo "Downloading $f"
        wget http://www.openslr.org/resources/11/$f -P $dl_dir/lm
    else
        echo "$f already downloaded"
    fi
done

if [ ! -f $dl_dir/lm/3-gram.pruned.1e-7.arpa ] ; then
    # Unzip the required .gz files without deleting them
    gunzip -c $dl_dir/lm/3-gram.pruned.1e-7.arpa.gz > $dl_dir/lm/3-gram.pruned.1e-7.arpa
fi
if [ ! -f $dl_dir/lm/4-gram.arpa ] ; then
    gunzip -c $dl_dir/lm/4-gram.arpa.gz > $dl_dir/lm/4-gram.arpa
fi
# gunzip -c $dl_dir/lm/3-gram.arpa.gz > $dl_dir/lm/3-gram.arpa
# gunzip -c $dl_dir/lm/4-gram.arpa.gz > $dl_dir/lm/4-gram.arpa
# gunzip -c $dl_dir/lm/librispeech-lm-norm.txt.gz > $dl_dir/lm/librispeech-lm-norm.txt

if [ ! -f $dl_dir/lm/G_3_gram.fst.txt ] ; then
    # Bring to kaldi fst format
    python -m kaldilm \
        --read-symbol-table=$word_txt_path \
        --disambig-symbol='#0' \
        --max-order=3 \
        $dl_dir/lm/3-gram.pruned.1e-7.arpa > $dl_dir/lm/G_3_gram.fst.txt  || exit 1;
fi


if [ ! -f $dl_dir/lm/G_4_gram.fst.txt ] ; then
    # Same for the 4-gram lm
    python -m kaldilm \
        --read-symbol-table=$word_txt_path \
        --disambig-symbol='#0' \
        --max-order=4 \
        $dl_dir/lm/4-gram.arpa > $dl_dir/lm/G_4_gram.fst.txt  || exit 1;
fi

python train_k2_decoder.py \
    hparams/train_tdnn_lstm_k2_dec.yaml \
    --data_folder=$data_folder \
    --musan_folder=$musan_folder \
    --bpe_dir=$bpe_dir \
    --token_type=$token_type \
    --bpe_model_path=$bpe_dir/"$n_bpe_units"-$token_type.model \
    --output_neurons=$n_bpe_units \
    --seed=$seed \
    --save_folder=$save_folder \
    --output_folder=$output_folder \
    --lm_path=$dl_dir/lm/G_3_gram.pt \
    --lm_path_4gram=$dl_dir/lm/G_4_gram.pt \
    --use_cuda=True \
    --device=$device \
    --train_splits=$train_splits