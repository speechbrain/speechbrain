#!/bin/bash

# Exit script if the incorrect number of arguments is provided
if [ $# -ne 2 ]
then
    echo "Usage: unzip_ksponspeech.sh <KsponSpeech dir> <dest dir>"
    exit 1
fi

KSPONPATH=$1
DESTPATH=$2

# Create directories for storing the unzipped files
mkdir -p "$DESTPATH/train"
mkdir -p "$DESTPATH/test"

echo "Expanding transcription"
unzip "$KSPONPATH/KsponSpeech_scripts.zip" -d "$DESTPATH" &

# Start unzipping training data in the background
echo "Expanding train data"
unzip "$KSPONPATH/KsponSpeech_01.zip" -d "$DESTPATH/train" &
unzip "$KSPONPATH/KsponSpeech_02.zip" -d "$DESTPATH/train" &
unzip "$KSPONPATH/KsponSpeech_03.zip" -d "$DESTPATH/train" &
unzip "$KSPONPATH/KsponSpeech_04.zip" -d "$DESTPATH/train" &
unzip "$KSPONPATH/KsponSpeech_05.zip" -d "$DESTPATH/train" &

# Wait for all background unzip processes to finish before proceeding
wait

echo "Expanding eval data"
unzip "$KSPONPATH/KsponSpeech_eval.zip" -d "$DESTPATH/test" &
wait # Wait again for the final unzip to complete

echo "All files have been expanded."
