"""
A helper class for reading the LJSpeech dataste

Authors
* Aleksandar Rachkov 2021
* Artem Ploujnikov 2021
"""

import csv
import os
import tempfile
from speechbrain.dataio.dataset import DynamicItemDataset


# TODO: Reduce repetition - this was originally copied from VCTK
class LJ:
    """
    A helper class for the loading of the LJ dataset

    Arguments
    ---------
    file_path:
        The path to the unzipped dataset

    """

    METADATA_COLS = ["file_name", "label_original", "label_normalized"]
    METADATA_FILE_NAME = "metadata.csv"
    DIR_WAV = "wavs"

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.wav_path = os.path.join(self.file_path, self.DIR_WAV)

    def _read_metadata_record(self, row):
        """
        Arguments
        ---------
        row: list
            a list of columns from the original metadata file

        Returns
        -------
        result: dict
            the metadata record, read as a dictionary
        """
        return dict(zip(self.METADATA_COLS, row))

    def _convert_metadata_record(self, record):
        """
        Converts the original metdata record to a

        Arguments
        ---------
        record: dict
            the original record

        Returns
        -------
        result: dict
            a record with keys corresponding to what will be
            used in a DynamicDataSet
        """
        return {
            "ID": record["file_name"],
            "speaker_id": 0,
            "wav": os.path.join(self.wav_path, f"{record['file_name']}.wav"),
            "label": record["label_normalized"],
            "label_original": record["label_original"],
        }

    @property
    def metadata_file(self):
        """
        Returns the full path to the metadata file
        """
        return os.path.join(self.file_path, self.METADATA_FILE_NAME)

    def _read_metadata(self):
        """
        Reads the metadata file

        Returns
        -------
        result: generator
            a generator of dictionaries
        """
        with open(self.metadata_file) as metadata_file:
            reader = csv.reader(
                metadata_file, delimiter="|", quoting=csv.QUOTE_NONE
            )
            for row in reader:
                yield self._read_metadata_record(row)

    def get_data(self):
        """
        Returns all available samples in the dataset

        Returns
        -------
        data: generator
            a generator of dictionaries with speaker data
        """
        return (
            self._convert_metadata_record(record)
            for record in self._read_metadata()
        )

    def to_csv(self, target):
        """
        Creates a CSV representation of the dataset

        Arguments
        ---------
        target: str or file
            a file name or a file-like object to which the script will be saved
        """
        if isinstance(target, str):
            with open(target, "w") as csv_file:
                self._write_csv(csv_file)
        else:
            self._write_csv(target)

    def _write_csv(self, csv_file):
        writer = csv.DictWriter(csv_file, fieldnames=self._get_csv_fieldnames())
        writer.writeheader()
        items = self.get_data()
        writer.writerows(item for item in items)

    def to_dataset(self):
        """
        Converts LJ to a SpeechBrain dataset

        Returns
        -------
        dataset: DynamicItemDataset
            A SpeechBrain dynamic dataset
        """
        with tempfile.NamedTemporaryFile("w") as csv_file:
            self.to_csv(csv_file)
            csv_file.flush()
            return DynamicItemDataset.from_csv(
                csv_file.name, raw_keys=["label_original"]
            )

    def _get_csv_fieldnames(self):
        """
        Returns the field names for the CSV file
        """
        sample_record = next(self.get_data())
        return sample_record.keys()


def load(file_path):
    """
    A convenience function to load an LJ dataset

    Arguments
    ---------
    file_path: str
        the path to the dataset

    Returns
    -------
    result: DynamicItemDataSet
        a processed dataset
    """

    return LJ(file_path).to_dataset()
