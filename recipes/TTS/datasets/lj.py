"""
A helper class for reading the LJSpeech dataste

Authors
* Aleksandar Rachkov 2021
* Artem Ploujnikov 2021
"""

import csv
import os
import shutil
import tempfile
from speechbrain.dataio.dataset import DynamicItemDataset


# TODO: Reduce repetition
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
    DIR_WAV = "wav"

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
        path = self.file_path
        _, _, filenames = next(os.walk(path))
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
            return DynamicItemDataset.from_csv(csv_file.name)

    def _get_csv_fieldnames(self):
        """
        Returns the field names for the CSV file
        """
        sample_record = next(self.get_data())
        return sample_record.keys()


def _get_fake_data():
    """
    Creates a LJ dataset from the included
    fake data for unit tests
    """
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, "mockdata", "lj")
    return LJ(data_path)


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


def test_to_csv():
    """
    Unit test for CSV creation
    """
    temp_dir = tempfile.mkdtemp()
    try:
        file_name = os.path.join(temp_dir, "test.csv")
        lj = _get_fake_data()
        lj.to_csv(file_name)
        with open(file_name) as csv_file:
            reader = csv.DictReader(csv_file)
            data = {row["ID"]: row for row in reader}
            item = data["LJ050-0159"]
            assert item["wav"].endswith("LJ050-0159.wav")
            assert item["label"].startswith("The Commission recommends")
            item = data["LJ050-0160"]
            assert item["wav"].endswith("LJ050-0160.wav")
            assert item["label"].startswith("The Commission further")
    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
