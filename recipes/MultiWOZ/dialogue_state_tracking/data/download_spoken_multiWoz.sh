# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

wget https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.zip
wget https://storage.googleapis.com/gresearch/dstc11/train.tts-verbatim.2022-07-27.txt
mv train.tts-verbatim.2022-07-27.txt train_manifest.txt
unzip train.tts-verbatim.2022-07-27.zip
# Using only one of the synthetic voices
mv train/tpa DSTC11_train_tts
wget https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.tts-verbatim.2022-07-27.zip
wget https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.human-verbatim.2022-09-29.zip
wget https://storage.googleapis.com/gresearch/dstc11/dev-dstc11.2022-07-27.txt
mv dev-dstc11.2022-07-27.txt dev_manifest.txt
unzip dev-dstc11.tts-verbatim.2022-07-27.zip
mv dev-dstc11.tts-verbatim DSTC11_dev_tts
unzip dev-dstc11.human-verbatim.2022-09-29.zip
mv dev-dstc11.human-verbatim DSTC11_dev_human
wget https://storage.googleapis.com/gresearch/dstc11/test-dstc11-tts-verbatim.2022-09-21.zip
wget https://storage.googleapis.com/gresearch/dstc11/test-dstc11.human-verbatim.2022-09-29.zip
# The test manifest with the annotations is already in the folder 
# because the one on the website does not have any annotations
# wget https://storage.googleapis.com/gresearch/dstc11/test-dstc11.2022-09-21.txt
# mv test-dstc11.2022-09-21.txt test_manifest.txt
unzip test-dstc11-tts-verbatim.2022-09-21.zip
mv tmp/tts DSTC11_test_tts
rm -d tmp
unzip test-dstc11.human-verbatim.2022-09-29.zip
mv test-dstc11.human-verbatim DSTC11_test_human