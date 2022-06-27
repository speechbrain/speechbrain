#!/bin/bash
cd data_proc/

# clone the IWSLT 2022 Tamasheq-French dataset
git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git

# train the tokenizer 
# it requires the command line extension for sentence piece, available here: https://github.com/google/sentencepiece
spm_train --input IWSLT2022_Tamasheq_data/taq_fra_clean/train/txt/train.fra  --vocab_size=1000 --model_type=unigram --model_prefix=IWSLT2022_Tamasheq_data/taq_fra_clean/train/spm_unigram1000

# generate jsons for speechbrain recipe
mkdir IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/
python to_json.py IWSLT2022_Tamasheq_data/taq_fra_clean/ IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/