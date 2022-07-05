#!/bin/bash
# ############################################################################
# Tamasheq-French data processing pipeline
#
# Requirements: python, git, sentencepiece command line extension
# Author:  Marcely Zanon Boito, 2022
# ############################################################################

cd data_proc/

# 1. clone the IWSLT 2022 Tamasheq-French dataset
git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git

# 2. train the tokenizer 
# /!\ it requires the command line extension for sentence piece, available here: https://github.com/google/sentencepiece
spm_train --input IWSLT2022_Tamasheq_data/taq_fra_clean/train/txt/train.fra  --vocab_size=1000 --model_type=unigram --model_prefix=IWSLT2022_Tamasheq_data/taq_fra_clean/train/spm_unigram1000

# 3. generate json files for the speechbrain recipe
mkdir IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/
python to_json.py IWSLT2022_Tamasheq_data/taq_fra_clean/ IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/