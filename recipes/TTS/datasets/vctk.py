# coding: utf-8
"""A helper class to read the VCTK dataset into a DynamicItemDataset.

Authors
* Artem Ploujnikov 2021
"""

import csv
import os
import re
import shutil
import tempfile
from glob import glob
from speechbrain.dataio.dataset import DynamicItemDataset


class VCTK:
    """
    A helper class for the loading of the VCTK dataset

    Arguments
    ----------
    file_path: str
        the path to the unzipped dataset

    """
    PATH_TXT = 'txt'
    PATH_WAV = 'wav48'
    PATH_SPEAKER_INFO = 'speaker-info.txt'
    PATTERN_TXT = '*.txt'
    RE_WHITESPACE = r'\s+'

    def __init__(self, file_path):
        self.file_path = file_path
        self._speakers = None

    @property
    def speaker_file_name(self):
        """
        Returns the speaker file name
        """
        return os.path.join(self.file_path, self.PATH_SPEAKER_INFO)

    @property
    def speakers(self):
        """
        Reads the speaker file associated with the dataset and returns
        as a dictionary mapping: {'<ID>': {'ID': '123', 'AGE': 25, ...}}
        """
        if self._speakers is None:
            self._load_speakers()
        return self._speakers

    def _load_speakers(self):
        """
        Loads the speaker file for the first time
        """
        self._speakers = {
            int(speaker['ID']): speaker
            for speaker in self._read_speaker_file()}

    def _read_speaker_file(self):
        """
        Reads the speaker file as an enumerable of
        dictionaries
        """
        with open(self.speaker_file_name) as speaker_file:
            speaker_file_iter = iter(speaker_file)
            line = next(speaker_file_iter)
            column_headers = list(
                filter(None, re.split(self.RE_WHITESPACE, line)))
            last_column = len(column_headers) - 1
            for line in speaker_file_iter:
                row = re.split(self.RE_WHITESPACE, line)
                # Note: The file is not tab-separated and the last
                # column may contain spaces
                values = row[:last_column] + [' '.join(row[last_column:])]
                values = [value.strip() for value in values]
                yield dict(zip(column_headers, values))


    def get_speaker_file_names(self, speaker_id):
        """
        Returns a list of (text_file, wav_file) tuples for the specified
        speaker ID

        Arguments
        ----------
        speaker_id: str
            the speaker ID

        Returns
        -------
        paths: list
            a list fo paths
        """
        txt_path, wav_path = self.get_speaker_paths(speaker_id)
        txt_file_pattern = os.path.join(txt_path, self.PATTERN_TXT)
        txt_files = glob(txt_file_pattern)
        return [(txt_file_name, _get_wav_file_name(wav_path, txt_file_name))
                for txt_file_name in txt_files]

    def get_speaker_paths(self, speaker_id):
        """
        Determines the paths to recordings and labels for the specified speaker ID

        Arguments
        ----------
        speaker_id: str
            the speaker ID

        Returns
        -------
        paths: tuple
            a (txt_path, wav_path) tuple with the paths to texts and paths to
            wave files, respectively
        """
        speaker_dir = f'p{speaker_id}'
        txt_path = os.path.join(self.file_path, self.PATH_TXT, speaker_dir)
        wav_path = os.path.join(self.file_path, self.PATH_WAV, speaker_dir)
        return txt_path, wav_path


    def get_all_speakers_data(self):
        """
        Returns data for all available speakers in the dataset

        Returns
        -------
        a generator of dictionaries with speaker data
        """
        # NOTE: The text is quick to read - it will be included in the CSV
        return (
            {'ID': filename_to_id(txt_file_name),
             'speaker_id': speaker_id,
             'speaker': speaker,
             'txt': txt_file_name,
             'wav': wav_file_name,
             'label': _read_text(txt_file_name)}
            for speaker_id, speaker in self.speakers.items()
            for txt_file_name, wav_file_name in self.get_speaker_file_names(speaker_id))

    def has_speaker_data(self, speaker_id):
        """
        Determines whether or not this particular dataset has data
        for the specified speaker. This is used primarily to run
        experiments on subsets of the original dataset

        Arguments
        ----------
        speaker_id
            the speaker ID to check

        Returns
        -------
        True if data for the speaker is available, False otherwise
        """
        txt_path, wav_path = self.get_speaker_paths(speaker_id)
        return os.path.isdir(txt_path) and os.path.isdir(wav_path)

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
        items = self.get_all_speakers_data()
        writer.writerows(
            _flatten_speaker(item) for item in items)


    def to_dataset(self):
        """
        Converts VCTK to a SpeechBrain dataset

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
        sample_record = next(self.get_all_speakers_data())
        return _flatten_speaker(sample_record).keys()

def _flatten_speaker(item):
    """
    Flattens the dataste data from
    {'speaker' : {'field1': 'value1', 'field2': 'value2},
     'txt': 'blah', 'wav': '1.wav'}
    to
    {'speaker_field1': 'value1', 'speaker_field2': 'value2',
     'txt': 'blah', 'wav': '1.wav'}

    Arguments
    ---------
    item: dict
        the flattened item
    """
    speaker_dict = {
        f'speaker_{key}': value for key, value in item['speaker'].items()}
    result = dict(item, **speaker_dict)
    del result['speaker']
    return result


def _read_text(file_name):
    """
    Reads the contents of a text file

    Arguments
    ---------

    file_name: str
        the name of the file to read

    Returns
    -------
    contents: str
        the file contents
    """
    with open(file_name) as text_file:
        return text_file.read().strip()


def _get_wav_file_name(target_path, file_name):
    """
    Computes the name of the .wav file corresponding to the specified
    text file. Used primarily to find the .wav file for a given .txt
    file based on the file convention used in the VCTK dataset

    Arguments
    ---------
    target_path: str
         the directory in which the file will be located
    file_name: str
        the original file name

    Returns
    --------
    full_path: str
        the full path to the wave file
    """
    base_name = os.path.basename(file_name)
    stripped_file_name, _ = os.path.splitext(base_name)
    return os.path.join(target_path, f'{stripped_file_name}.wav')


def filename_to_id(file_name):
    """
    Returns the provided file name without the extension
    and the directory part. Based on the convention of
    the dataset, it can be used as an ID

    Arguments
    ---------
    file_name: str
        the file name (of the .txt or .wav file)

    Returns
    -------
    item_id: str
        the ID part of the filename
    """
    base_name = os.path.basename(file_name)
    item_id, _ = os.path.splitext(base_name)
    return item_id


def _get_fake_data():
    """
    Creates a VCTK dataset from the included
    fake data for unit tests
    """
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, 'testdata', 'fake_vctk')
    return VCTK(data_path)


def test_speakers():
    """
    Unit test for getting a list of speakers
    """
    vctk = _get_fake_data()
    assert 225 in vctk.speakers
    speaker = vctk.speakers[225]
    assert speaker.get('GENDER') == 'F'
    assert speaker.get('AGE') == '23'
    assert speaker.get('REGION') == 'Southern England'
    speaker = vctk.speakers[226]
    assert speaker.get('GENDER') == 'M'
    assert speaker.get('AGE') == '22'
    assert speaker.get('REGION') == 'Surrey'
    assert len(vctk.speakers) == 2


def test_get_all_speakers_data():
    """
    Unit test for the retrieval of all speakers' data
    """
    vctk = _get_fake_data()
    data = list(vctk.get_all_speakers_data())
    assert any(
        item['speaker']['ID'] == '225'
        for item in data)
    assert any(
        item['speaker']['ID'] == '226'
        for item in data)
    assert any(
        item['wav'].endswith('p225/p225_002.wav')
        for item in data)
    assert any(
        item['wav'].endswith('p226/p226_003.wav')
        for item in data)
    assert any(
        item['txt'].endswith('p225/p225_002.txt')
        for item in data)
    assert all(
        os.path.exists(item['wav'])
        and os.path.exists(item['txt'])
        for item in data)

def test_to_csv():
    """
    Unit test for CSV creation
    """
    temp_dir = tempfile.mkdtemp()
    try:
        file_name = os.path.join(temp_dir, 'test.csv')
        vctk = _get_fake_data()
        vctk.to_csv(file_name)
        with open(file_name) as csv_file:
            reader = csv.DictReader(csv_file)
            data = {row['ID']: row for row in reader}
            item = data['p225_001']
            assert item['speaker_ID'] == '225'
            assert item['txt_file_name'].endswith('p225_001.txt')
            assert item['wav'].endswith('p225_001.wav')
            assert item['label'] == 'Please call Stella.'
            item = data['p226_002']
            assert item['speaker_ID'] == '226'
            assert item['txt_file_name'].endswith('p226_002.txt')
            assert item['wav'].endswith('p226_002.wav')
            assert item['label'] == 'Ask her to bring these things with her from the store.'
    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
