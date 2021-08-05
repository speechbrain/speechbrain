#!/usr/bin/env bash

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0

# Example of how to combine k2 with speechbrain pretrained encoder to decode well and fastly. 
set -eou pipefail

libri_dirs=(
#/root/fangjun/data/librispeech/LibriSpeech
#/export/corpora5/LibriSpeech
#/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech
#/export/common/data/corpora/ASR/openslr/SLR12/LibriSpeech
/kome/luomingshuang/audio-data/librispeech/LibriSpeech
)

libri_dir=
for d in ${libri_dirs[@]}; do
  if [ -d $d ]; then
    libri_dir=$d
    break
  fi
done

if [ ! -d $libri_dir ]; then
  echo "Please set LibriSpeech dataset path before running this script"
  exit 1
fi

echo "LibriSpeech dataset dir: $libri_dir"

stage=6

if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.org/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  python utils/download_speechbrain_pretrained_models.py
fi

if [ $stage -le 3 ]; then
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
fi

if [ $stage -le 4 ]; then
  local/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_nosp \
    "<UNK>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 5 ]; then
  # Build G
  
  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=1 \
    data/local/lm/lm_tgmed.arpa >data/lang_nosp/G_uni.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/local/lm/lm_tgmed.arpa >data/lang_nosp/G.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=4 \
    data/local/lm/lm_fglarge.arpa >data/lang_nosp/G_4_gram.fst.txt
  
  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi

if [ $stage -le 6 ]; then
  python3 utils/build_HLG.py
  python3 utils/convert_G_4_gram_fst.py
fi

if [ $stage -le 7 ]; then
  python3 ./test_k2_speechbrain_HLG.py \
  --use-lm-score=True \
  --use-whole-lattice=True \
fi
