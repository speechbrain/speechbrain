#!/bin/bash

# Iterate over all .gz files in libriheavy directory
for gz_file in /home/adelmou/proj/libriheavy/*.gz; do
    # Get the filename without path and extension
    base_name=$(basename "$gz_file" .jsonl.gz)
    
    # Create output CSV path 
    output_csv="/home/adelmou/proj/libriheavy/${base_name}.csv"
 
    # Run extract_transcripts.py on each file
    python /home/adelmou/proj/speechbrain/speechbrain/templates/spoken_language_modeling/extract_transcripts.py \
        --input "$gz_file" \
        --output "$output_csv"
done
