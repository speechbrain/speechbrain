import argparse
import json
import csv
import gzip
import logging
from pathlib import Path
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="""The jsonl file of libriheavy a jsonl.gz file.
        """,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="The output CSV file.",
    )
    return parser.parse_args()


def extract_transcripts(jsonl_gz_file_path, csv_file_path):
    print(f'Extracting transcriptions from {jsonl_gz_file_path} to {csv_file_path}')
    # Open the gzipped JSONL file to count lines
    with gzip.open(jsonl_gz_file_path, 'rt') as jsonl_file:
        total_lines = sum(1 for _ in jsonl_file)
    
    # Open the gzipped JSONL file and the CSV file
    with gzip.open(jsonl_gz_file_path, 'rt') as jsonl_file, open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header to the CSV file
        csv_writer.writerow(['Recording ID', 'Start Time', 'Duration', 'Transcription Normalized'])
        
        # Initialize the progress bar
        for line in tqdm(jsonl_file, total=total_lines, desc="Processing lines"):
            data = json.loads(line)
            recording_id = data['recording']['id']
            start_time = data['start']
            duration = data['duration']
            texts = data['supervisions'][0]['custom']['texts']
            # Extract transcriptions
            transcription_norm = texts[1]
            # Write the row to the CSV file
            csv_writer.writerow([recording_id, start_time, duration, transcription_norm])

    print(f'CSV file has been created at {csv_file_path}')


def main():
    args = get_args()
    ifile = args.input
    assert ifile.is_file(), f"File not exists : {ifile}"
    assert str(ifile).endswith("jsonl.gz"), f"Expect a jsonl gz file, given : {ifile}"
    ofile = args.output
    assert not ofile.is_file(), f"File already exists : {ofile}"
    os.makedirs(ofile.parent, exist_ok=True)
    extract_transcripts(ifile, ofile)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()