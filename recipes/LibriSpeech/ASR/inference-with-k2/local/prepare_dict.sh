#!/usr/bin/env bash

# Copyright
# Apache 

# Prepare the dictionary

stage=1

if [ $# -ne 2 ]; then
  echo "Usage: $0 <lm-dir> <dst-dir>"
  echo "e.g.: /export/a15/vpanayotov/data/lm data/local/dict"
  exit 1
fi

lm_dir=$1
dst_dir=$2

vocab=$lm_dir/librispeech-vocab.txt
[ ! -f $vocab ] && echo "$0: vocabulary file not found at $vocab" && exit 1;

lexicon_raw_nosil=$dst_dir/lexicon_raw_nosil.txt

mkdir -p $dst_dir || exit 1;

if [[ ! -s "$lexicon_raw_nosil" ]]; then
  cp $lm_dir/librispeech-bpe-lexicon.txt $lexicon_raw_nosil || exit 1;
fi

if [ $stage -le 1 ]; then
  silence_phones=$dst_dir/silence_phones.txt
  optional_silence=$dst_dir/optional_silence.txt
  
  echo "Preparing phone lists"
  (echo SIL; echo SPN;) > $silence_phones
  echo SIL > $optional_silence
fi

if [ $stage -le 3 ]; then
  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |\
  cat - $lexicon_raw_nosil | sort | uniq >$dst_dir/lexicon.txt
  echo "Lexicon text file saved as: $dst_dir/lexicon.txt"
fi
