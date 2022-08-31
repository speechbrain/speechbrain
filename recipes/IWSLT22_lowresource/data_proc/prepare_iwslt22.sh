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


# 2. generate json files for the speechbrain recipe
mkdir IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/
python prepare_iwslt22.py IWSLT2022_Tamasheq_data/taq_fra_clean/ IWSLT2022_Tamasheq_data/taq_fra_clean/json_version/