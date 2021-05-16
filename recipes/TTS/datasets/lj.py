import csv
import os
import shutil
import tempfile
from speechbrain.dataio.dataset import DynamicItemDataset

from .common import filename_to_id

# TODO: Review the use of this class


class LJ:
    """
    A helper class for the loading of the LJ dataset

    Arguments
    ---------
    file_path:
        The path to the unzipped dataset

    """

    def __init__(self, file_path: str):
        self.file_path = file_path

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
        return(
            {'ID': filename_to_id(wav_file_name),
             "wav": wav_file_name}
            for wav_file_name in
            [os.path.join(path, f) for f in filenames])

    def to_csv(self, target):
        """
        Creates a CSV representation of the dataset

        Arguments
        ---------
        target: str or file
            a file name or a file-like object to which the script will be saved
        """
        if isinstance(target, str):
            with open(target, 'w') as csv_file:
                self._write_csv(csv_file)
        else:
            self._write_csv(target)

    def _write_csv(self, csv_file):
        writer = csv.DictWriter(
            csv_file,
            fieldnames=self._get_csv_fieldnames())
        writer.writeheader()
        items = self.get_data()
        writer.writerows(
            item for item in items)

    def to_dataset(self):
        """
        Converts LJ to a SpeechBrain dataset

        Returns
        -------
        dataset: DynamicItemDataset
            A SpeechBrain dynamic dataset
        """
        with tempfile.NamedTemporaryFile('w') as csv_file:
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
    Creates a VCTK dataset from the included
    fake data for unit tests
    """
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, 'mockdata', 'lj')
    return LJ(data_path)


def test_to_csv():
    """
    Unit test for CSV creation
    """
    temp_dir = tempfile.mkdtemp()
    try:
        file_name = os.path.join(temp_dir, 'test.csv')
        lj = _get_fake_data()
        lj.to_csv(file_name)
        with open(file_name) as csv_file:
            reader = csv.DictReader(csv_file)
            data = {row['ID']: row for row in reader}
            item = data['LJ050-0159']
            assert item['wav'].endswith('LJ050-0159.wav')
            item = data['LJ050-0160']
            assert item['wav'].endswith('LJ050-0160.wav')
    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
