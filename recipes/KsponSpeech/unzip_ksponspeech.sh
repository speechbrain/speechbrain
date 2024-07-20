#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: unzip_ksponspeech.sh <KsponSpeech dir> <dest dir>"
fi

KSPONPATH=$1
DESTPATH=$2

mkdir -p $DESTPATH/train
mkdir -p $DESTPATH/test

echo "expanding transcription"
unzip "$KSPONPATH/전시문_통합_스크립트/KsponSpeech_scripts.zip" -d $DESTPATH

echo "expanding train data"
unzip "$KSPONPATH/한국어_음성_분야/KsponSpeech_01.zip" -d $DESTPATH/train
unzip "$KSPONPATH/한국어_음성_분야/KsponSpeech_02.zip" -d $DESTPATH/train
unzip "$KSPONPATH/한국어_음성_분야/KsponSpeech_03.zip" -d $DESTPATH/train
unzip "$KSPONPATH/한국어_음성_분야/KsponSpeech_04.zip" -d $DESTPATH/train
unzip "$KSPONPATH/한국어_음성_분야/KsponSpeech_05.zip" -d $DESTPATH/train

echo "expanding eval data"
unzip "$KSPONPATH/평가용_데이터/KsponSpeech_eval.zip" -d $DESTPATH/test