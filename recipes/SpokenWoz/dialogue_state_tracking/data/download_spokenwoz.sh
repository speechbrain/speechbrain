# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

# Fetching files from the web
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_train_dev.tar.gz
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_train_dev.tar.gz
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/audio_5700_test.tar.gz
wget https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/text_5700_test.tar.gz
# Decompressing files (excluding MacOS files)
tar -xzf audio_5700_train_dev.tar.gz --exclude '\._*'
tar -xzf text_5700_train_dev.tar.gz --exclude '\._*'
tar -xzf audio_5700_test.tar.gz --exclude '\._*' 
tar -xzf text_5700_test.tar.gz --exclude '\._*'
