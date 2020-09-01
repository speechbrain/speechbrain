"""
Data i/o operations.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Ju-Chieh Chou 2020
"""

import os
import re
import csv
import torch
import psutil
import random
import pickle
import logging
import hashlib
import numpy as np
import soundfile as sf
import multiprocessing as mp
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DataLoaderFactory(torch.nn.Module):
    """
    Creates data loaders for a csv file

    Arguments
    ---------
    csv : str
        the csv file that itemizes the data
    batch_size : int, optional
        Default: 1 .  The data itemized in the csv file are automatically
        organized in batches. In the case of variable size tensors, zero
        padding is performed. When batch_size=1, the data are simply processed
        one by one without the creation of batches.
    csv_read : list, None, optional
        Default: None .  A list of data entries may be specified.  If
        specified, only those data entries are read from the csv file. If None,
        read all the data entries.
    sentence_sorting : {'ascending', 'descending', 'random', 'original'}
        Default: 'original'. This parameter specifies how to sort the data
        before the batch creation. Ascending and descending values sort the
        data using the "duration" field in the csv files. Random sort the data
        randomly, while original (the default option) keeps the original
        sequence of data defined in the csv file. Note that this option affects
        the batch creation. If the data are sorted in ascending or descending
        order the batches will approximately have the same size and the need
        for zero padding is minimized. Instead, if sentence_sorting is set to
        random, the batches might be composed of both short and long sequences
        and several zeros might be added in the batch. When possible, it is
        desirable to sort the data. This way, we use more efficiently the
        computational resources, without wasting time on processing time steps
        composed on zeros only. Note that is the data are sorted in ascending/
        descending errors the same batches will be created every time we want
        to loop over the dataset, while if we set a random order the batches
        will be different every time we loop over the dataset.
    num_workers : int, optional
        Default: 0 . Data is read using the pytorch data_loader.  This option
        sets the number of workers used to read the data from disk and form the
        related batch of data. Please, see the pytorch documentation on the
        data loader for more details.
    cache : bool, optional
        Default: False . When set to true, this option stores the input data in
        a variable called self.cache (see DataLoaderFactory in data_io.py). In
        practice, the first time the data are read from the disk, they are
        stored in the cpu RAM. If the data needs to be used again (e.g. when
        loops>1) the data will be read from the RAM directly.  If False, data
        are read from the disk every time.  Data are stored until a certain
        percentage of the total ram available is reached (see cache_ram_percent
        below).
    cache_ram_percent : float, optional
        Default: 75 . If cache if True, data will be stored in the cpu RAM
        until the total RAM occupation is less or equal than the specified
        threshold (by default 75%). In practice, if a lot of RAM is available
        several data will be stored in memory, otherwise, most of them will be
        read from the disk directly.
    select_n_setences : int, optional
        Default: None . It selects the first N sentences of the CSV file.
    avoid_if_longer_than : float, optional
        Default: 36000 . It excludes sentences longer than the specified value
        in seconds.
    avoid_if_shorter_than : float, optional
        Default: 0 . It excludes sentences shorter than the specified value in
        seconds.
    drop_last : bool, optional
        Default: False . This is an option directly passed to the pytorch
        dataloader (see the related documentation for more details). When True,
        it skips the last batch of data if contains fewer samples than the
        other ones.
    padding_value : int, optional
        Default: 0. Value to use for padding.
    add_padding_label : bool, optional
        Default: False. If set to True, the padding value will be add to the label dict as an additional label
    replacements : dict, optional
        String replacements to perform in this method
    output_folder : str, optional
        A folder for storing the label dict

    Example
    -------
    >>> csv_file = 'samples/audio_samples/csv_example2.csv'
    >>> # Initialization of the class
    >>> data_loader=DataLoaderFactory(csv_file)
    >>> # When called, creates a dataloader for each entry in the csv file
    >>> # The sample has two: wav and spk
    """

    def __init__(
        self,
        csv_file,
        batch_size=1,
        csv_read=None,
        sentence_sorting="random",
        num_workers=0,
        cache=False,
        cache_ram_percent=75,
        select_n_sentences=None,
        avoid_if_longer_than=36000,
        avoid_if_shorter_than=0,
        drop_last=False,
        padding_value=0,
        add_padding_label=False,
        replacements={},
        output_folder=None,
        label_parsing_func=None,
    ):
        super().__init__()

        # Store init params
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.csv_read = csv_read
        self.sentence_sorting = sentence_sorting
        self.num_workers = num_workers
        self.cache = cache
        self.cache_ram_percent = cache_ram_percent
        self.select_n_sentences = select_n_sentences
        self.avoid_if_longer_than = avoid_if_longer_than
        self.avoid_if_shorter_than = avoid_if_shorter_than
        self.drop_last = drop_last
        self.padding_value = padding_value
        self.replacements = replacements
        self.output_folder = output_folder
        self.label_parsing_func = label_parsing_func
        self.add_padding_lab = add_padding_label

        # Other variables
        self.supported_formats = self.get_supported_formats()

        # Shuffle the data every time if random is selected
        if self.sentence_sorting == "random":
            self.shuffle = True
        else:
            self.shuffle = False

    def forward(self):
        """
        Output:
        dataloader: It is a list returning all the dataloaders created.
        """

        # create data dictionary
        data_dict = self.generate_data_dict()

        self.label_dict = self.label_dict_creation(data_dict)

        self.data_len = len(data_dict["data_list"])

        if self.csv_read is None:
            self.csv_read = data_dict["data_entries"]

        # Creating a dataloader
        dataset = DatasetFactory(
            data_dict,
            self.label_dict,
            self.supported_formats,
            self.csv_read,
            self.cache,
            self.cache_ram_percent,
            self.label_parsing_func,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=self.batch_creation,
        )

        return self.dataloader

    def batch_creation(self, data_lists):
        """
        Data batching

        When necessary this performs zero padding on the input tensors. The
        function is executed in collate_fn of the pytorch DataLoader.

        Arguments
        ---------
        data_list : list
            list of data returned by the data reader [data_id, data, data_len]

        Returns
        -------
        list :
            list containing the final batches:
            [data_id,data,data_len] where zero-padding is
            performed where needed.
        """
        batch = []

        n_data_entries = len(data_lists[0])

        batch_list = [[[] for i in range(3)] for j in range(n_data_entries)]

        for data_entry in data_lists:
            for i, data in enumerate(data_entry):
                batch_list[i][0].append(data[0])
                batch_list[i][1].append(data[1])
                batch_list[i][2].append(data[2])

        for data_list in batch_list:

            # Convert all to torch tensors
            self.numpy2torch(data_list)

            # Save sentence IDs
            snt_ids = data_list[0]

            # Save duration
            sequences = data_list[1]
            time_steps = torch.tensor(data_list[2])

            # Check if current element is a tensor
            if isinstance(sequences[0], torch.Tensor):

                # Padding the sequence of sentences (if needed)
                batch_data = self.padding(sequences)

                # Return % of time steps without padding (useful for save_batch)
                time_steps = time_steps / batch_data.shape[1]

            else:
                # Non-tensor case
                batch_data = sequences

            # Batch composition
            batch.append([snt_ids, batch_data, time_steps])

        return batch

    def padding(self, sequences):
        """
        This function perform zero padding on the input list of tensors

        Arguments
        ---------
        sequences : list
            the list of tensors to pad

        Returns
        -------
        torch.tensor
            a tensor gathering all the padded tensors

        Example
        -------
        >>> csv_file = 'samples/audio_samples/csv_example2.csv'
        >>> # Initialization of the class
        >>> data_loader=DataLoaderFactory(csv_file)
        >>> # list of tensors
        >>> tensor_lst=[torch.tensor([1,2,3,4]),torch.tensor([1,2])]
        >>> data_loader.padding(tensor_lst)[1,:]
        tensor([1., 2., 0., 0.])
        """

        # Batch size
        batch_size = len(sequences)

        # Computing data dimensionality (only the first time)
        try:
            self.data_dim
        except Exception:
            self.data_dim = list(sequences[0].shape[2:])

        # Finding the max len across sequences
        max_len = max([s.size(0) for s in sequences])

        # Batch out dimensions
        out_dims = [batch_size] + [max_len] + self.data_dim

        # Batch initialization
        batch_data = torch.zeros(out_dims) + self.padding_value

        # Appending data
        for i, tensor in enumerate(sequences):
            length = tensor.shape[0]
            batch_data[i, :length, ...] = tensor

        return batch_data

    def numpy2torch(self, data_list):
        """
        This function coverts a list of numpy tensors to torch.Tensor

        Arguments
        ---------
        data_list : list
            list of numpy arrays

        Returns
        -------
        None
            The tensors are modified in place!

        Note
        ----
        Did you notice, the tensors are modified in place!

        Example
        -------
        >>> csv_file = 'samples/audio_samples/csv_example2.csv'
        >>> # Initialization of the class
        >>> data_loader=DataLoaderFactory(csv_file)
        >>> # list of numpy tensors
        >>> tensor_lst=[[np.asarray([1,2,3,4]),np.asarray([1,2])]]
        >>> # Applying zero padding
        >>> data_loader.numpy2torch(tensor_lst)
        >>> print(tensor_lst)
        [[tensor([1, 2, 3, 4]), tensor([1, 2])]]
        >>> print(type(tensor_lst[0][0]))
        <class 'torch.Tensor'>
        """

        # Covert all the elements of the list to torch.Tensor
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                if isinstance(data_list[i][j], np.ndarray):
                    data_list[i][j] = torch.from_numpy(data_list[i][j])

    def label_dict_creation(self, data_dict):
        # create label counts and label2index automatically when needed
        label_dict = {}
        if self.output_folder is not None:
            label_dict_file = os.path.join(self.output_folder, "label_dict.pkl")

            # Read previously stored label_dict
            if os.path.isfile(label_dict_file):
                label_dict = load_pkl(label_dict_file)

        # Update label dict
        for snt in data_dict:
            if not isinstance(data_dict[snt], dict):
                continue

            for elem in data_dict[snt]:
                if "format" not in data_dict[snt][elem]:
                    continue

                count_lab, labels = self._should_count(data_dict[snt][elem])

                if not count_lab:
                    continue

                if elem not in label_dict:
                    label_dict[elem] = {}
                    label_dict[elem]["counts"] = {}

                for lab in labels:
                    if lab not in label_dict[elem]["counts"]:
                        label_dict[elem]["counts"][lab] = 1
                    else:
                        label_dict[elem]["counts"][lab] += 1

        # create label2index:
        for lab in label_dict:
            # sorted_ids = sorted(label_dict[lab]["counts"].keys())
            cnt_id = -1

            label_dict[lab]["lab2index"] = {}
            label_dict[lab]["index2lab"] = {}

            # append <pad> token to label_dict
            if self.add_padding_lab:
                label_dict[lab]["lab2index"]["<pad>"] = self.padding_value
                label_dict[lab]["index2lab"][self.padding_value] = "<pad>"

            for lab_id in label_dict[lab]["counts"]:
                if (
                    cnt_id == int(self.padding_value) - 1
                    and self.add_padding_lab
                ):
                    cnt_id = cnt_id + 2
                else:
                    cnt_id = cnt_id + 1
                label_dict[lab]["lab2index"][lab_id] = cnt_id
                label_dict[lab]["index2lab"][cnt_id] = lab_id

        # saving the label_dict:
        if self.output_folder is not None:
            save_pkl(label_dict, label_dict_file)

        return label_dict

    def _should_count(self, data_dict_elem):
        """Compute whether this label should be counted or not"""
        count_lab = False
        labels = []
        opts = data_dict_elem["options"]

        if "label_func" in opts and opts["label_func"] == "True":
            assert self.label_parsing_func, (
                "A parsing function must be defined for "
                "the labels and passed to DataLoadFactory"
            )
            labels = self.label_parsing_func(data_dict_elem["data"])
            count_lab = False

        if (
            data_dict_elem["format"] == "string"
            and ("label" not in opts or opts["label"] == "True")
            and (not ("label_func" in opts and opts["label_func"] == "True"))
        ):

            if len(data_dict_elem["data"].split(" ")) > 1:
                # Processing list of string labels
                labels = data_dict_elem["data"].split(" ")
                count_lab = True

            else:
                # Processing a single label
                labels = [data_dict_elem["data"]]
                count_lab = True

        if data_dict_elem["format"] == "pkl":

            labels = load_pkl(data_dict_elem["data"])

            # Create counts if tensor is a list of integers
            if isinstance(labels, list):
                if isinstance(labels[0], int):
                    count_lab = True

            if isinstance(labels, np.ndarray):
                if "numpy.int" in str(type(labels[0])):
                    count_lab = True

            # Create counts if tensor is a list of integers
            if isinstance(labels, torch.Tensor):

                if labels.type() == "torch.LongTensor":
                    count_lab = True
                if labels.type() == "torch.IntTensor":
                    count_lab = True

        return count_lab, labels

    def generate_data_dict(self):
        """
        Create a dictionary from the csv file

        Returns
        -------
        dict
            Dictionary with the data itemized in the csv file

        Example
        -------
        >>> csv_file = 'samples/audio_samples/csv_example2.csv'
        >>> data_loader=DataLoaderFactory(csv_file)
        >>> data_loader.generate_data_dict().keys()
        dict_keys(['example1', 'data_list', 'data_entries'])
        """
        # Initial prints
        logger.debug("Creating dataloader for %s" % (self.csv_file))

        # Initialization of the data_dict
        data_dict = {}

        # CSV file reader
        reader = csv.reader(open(self.csv_file, "r"))

        first_row = True

        # Tracking the total number of sentences and their duration
        total_duration = 0
        total_sentences = 0

        for row in reader:

            # Skipping empty lines
            if len(row) == 0:
                continue

            # remove left/right spaces
            row = [elem.strip(" ") for elem in row]

            # update sentence counter
            total_sentences = total_sentences + 1
            if self.select_n_sentences is not None:
                if total_sentences > self.select_n_sentences:
                    break

            # Check and get field list from first row
            if first_row:
                self._check_first_row(row)
                field_lst = row
                first_row = False
                continue

            # replace local variables with global ones
            variable_finder = re.compile(r"\$[\w.]+")
            for i, item in enumerate(row):
                try:
                    row[i] = variable_finder.sub(
                        lambda x: self.replacements[x[0]], item,
                    )
                except KeyError as e:
                    e.args = (
                        *e.args,
                        "The item '%s' contains variables "
                        "not included in 'replacements'" % item,
                    )
                    raise

            # Make sure that the current row contains all the fields
            if len(row) != len(field_lst):
                err_msg = (
                    'The row "%s" of the cvs file %s does not '
                    "contain the right number fields (they must be %i "
                    "%s"
                    ")" % (row, self.csv_file, len(field_lst), field_lst)
                )
                raise ValueError(err_msg)

            # Filling the data dictionary
            for i, field in enumerate(field_lst):
                if i == 0:
                    row_id = row[i]
                    data_dict[row_id] = {}
                    continue

                elif field == "duration":
                    data_dict[row_id][field] = row[i]
                    total_duration = total_duration + float(row[i])
                    continue

                format_field = field.endswith("_format")
                opts_field = field.endswith("_opts")
                field = field.replace("_format", "").replace("_opts", "")

                if field not in data_dict[row_id]:
                    data_dict[row_id][field] = {
                        "data": {},
                        "format": {},
                        "options": {},
                    }

                if not format_field and not opts_field:
                    data_dict[row_id][field]["data"] = row[i]

                elif format_field:
                    data_dict[row_id][field]["format"] = row[i]

                elif opts_field:
                    data_dict[row_id][field]["options"] = self._parse_opts(
                        row[i]
                    )

            # Avoiding sentence that are too long or too short
            self._avoid_short_long_sentences(data_dict, row_id)

        data_dict = self.sort_sentences(data_dict, self.sentence_sorting)

        logger.debug("Number of sentences: %i" % (len(data_dict.keys())))
        logger.debug("Total duration (hours): %1.2f" % (total_duration / 3600))
        logger.debug(
            "Average duration (seconds): %1.2f"
            % (total_duration / len(data_dict.keys()))
        )

        # Adding sorted list of sentences
        data_dict["data_list"] = list(data_dict.keys())

        snt_ex = data_dict["data_list"][0]

        data_entries = list(data_dict[snt_ex].keys())
        data_entries.remove("duration")

        data_dict["data_entries"] = data_entries

        return data_dict

    def _avoid_short_long_sentences(self, snt, row_id):
        """Excludes long and short sentences from the dataset"""
        if float(snt[row_id]["duration"]) > self.avoid_if_longer_than:
            del snt[row_id]

        elif float(snt[row_id]["duration"]) < self.avoid_if_shorter_than:
            del snt[row_id]

    def _check_first_row(self, row):
        # Make sure ID field exists
        if "ID" not in row:
            err_msg = (
                "The mandatory field ID (i.e, the field that contains "
                "a unique  id for each sentence is not present in the "
                "csv file  %s" % (self.csv_file)
            )
            raise ValueError(err_msg)

        # Make sure the duration field exists
        if "duration" not in row:
            err_msg = (
                "The mandatory field duration (i.e, the field that "
                "contains the  duration of each sentence is not "
                "present in the csv  file %s" % (self.csv_file)
            )
            raise ValueError(err_msg)

        if len(row) == 2:
            err_msg = (
                "The cvs file %s does not contain features entries! "
                "The features are specified with the following fields:"
                "feaname, feaname_format, feaname_opts" % (self.csv_file)
            )
            raise ValueError(err_msg)

        # Make sure the features are expressed in the following way:
        # feaname, feaname_format, feaname_opts
        feats = row[2:]
        feat_names = feats[0::3]

        for feat_name in feat_names:

            if feat_name + "_format" not in row:
                err_msg = (
                    "The feature %s in the cvs file %s does not "
                    "contain the field %s to specified its format."
                    % (feat_name, self.csv_file, feat_name + "_format")
                )
                raise ValueError(err_msg)

            if feat_name + "_opts" not in row:
                err_msg = (
                    "The feature %s in the cvs file %s does not "
                    "contain the field %s to specified the reader "
                    "options. "
                    % (feat_name, self.csv_file, feat_name + "_opts")
                )
                raise ValueError(err_msg)

    def _parse_opts(self, entry):
        """Parse options from a list of options"""
        if len(entry) == 0:
            return {}

        opts = {}
        for opt in entry.split(" "):
            opt_name, opt_val = opt.split(":")
            opts[opt_name] = opt_val

        return opts

    @staticmethod
    def sort_sentences(data_dict, sorting):
        """
        Sort the data dictionary

        Arguments
        ---------
        data_dict : dict
            Dictionary with the data itemized in the csv file
        sorting : {'ascending', 'descending', 'random', 'original'}
            Default: 'original'. This parameter specifies how to sort the data
            before the batch creation. Ascending and descending values sort the
            data using the "duration" field in the csv files. Random sort the data
            randomly, while original (the default option) keeps the original
            sequence of data defined in the csv file. Note that this option affects
            the batch creation. If the data are sorted in ascending or descending
            order the batches will approximately have the same size and the need
            for zero padding is minimized. Instead, if sentence_sorting is set to
            random, the batches might be composed of both short and long sequences
            and several zeros might be added in the batch. When possible, it is
            desirable to sort the data. This way, we use more efficiently the
            computational resources, without wasting time on processing time steps
            composed on zeros only. Note that is the data are sorted in ascending/
            descending errors the same batches will be created every time we want
            to loop over the dataset, while if we set a random order the batches
            will be different every time we loop over the dataset.


        Returns
        -------
        dict
            dictionary with the sorted data
        """
        # Initialization of the dictionary
        # Note: in Python 3.7 the order of the keys added in the dictionary is
        # preserved
        sorted_dictionary = {}

        # Ascending sorting
        if sorting == "ascending":
            sorted_ids = sorted(
                sorted(data_dict.keys()),
                key=lambda k: float(data_dict[k]["duration"]),
            )

        # Descending sorting
        if sorting == "descending":
            sorted_ids = sorted(
                sorted(data_dict.keys()),
                key=lambda k: -float(data_dict[k]["duration"]),
            )

        # Original order
        if sorting == "original" or sorting == "random":
            sorted_ids = list(data_dict.keys())

        # Filling the dictionary
        for snt_id in sorted_ids:
            sorted_dictionary[snt_id] = data_dict[snt_id]

        return sorted_dictionary

    @staticmethod
    def get_supported_formats():
        """
        Itemize the supported reading formats

        Returns
        -------
        dict
            The supported formats as top level keys, and the related
            readers plus descriptions in nested keys.

        Example
        -------

        >>> data_loader=DataLoaderFactory(
        ...     csv_file='samples/audio_samples/csv_example2.csv'
        ... )
        >>> data_loader.get_supported_formats()['flac']['description']
        'FLAC...'
        """

        # Initializing the supported formats dictionary
        supported_formats = {}

        # Getting the soundfile formats
        sf_formats = sf.available_formats()

        # Adding soundfile formats
        for wav_format in sf_formats.keys():
            wav_format = wav_format.lower()
            supported_formats[wav_format] = {}
            supported_formats[wav_format]["reader"] = read_wav_soundfile
            supported_formats[wav_format]["description"] = sf_formats[
                wav_format.upper()
            ]
            supported_formats[wav_format]["file"] = True

        # Adding the other supported formats
        supported_formats["pkl"] = {}
        supported_formats["pkl"]["reader"] = read_pkl
        supported_formats["pkl"]["description"] = "Python binary format"
        supported_formats["pkl"]["file"] = True

        supported_formats["string"] = {}
        supported_formats["string"]["reader"] = read_string
        supported_formats["string"]["description"] = "Plain text"
        supported_formats["string"]["file"] = False

        return supported_formats


class DatasetFactory(Dataset):
    """
    This class implements the dataset needed by the pytorch data_loader.

    Arguments
    ---------
    data_dict : dict
        a dictionary containing all the entries of the csv file.
    supported_formats : dict
        a dictionary contaning the reading supported format
    data_entries : list
        it is a list containing the data_entries to read from the csv file
    do_cache : bool, optional
        Default:False When set to true, this option stores the input data in a
        variable called self.cache. In practice, the first time the data are
        read from the disk, they are stored in the cpu RAM. If the data needs
        to be used again (e.g. when loops>1) the data will be read from the RAM
        directly. If False, data are read from the disk every time.  Data are
        stored until a certain percentage of the total ram available is reached
        (see cache_ram_percent below)
    cache_ram_percent : float
        If cache if True, data will be stored in the cpu RAM until the total
        RAM occupation is less or equal than the specified threshold. In
        practice, if a lot of RAM is available several  data will be stored in
        memory, otherwise, most of them will be read from the disk directly.


    Example
    -------
    >>> csv_file = 'samples/audio_samples/csv_example2.csv'
    >>> data_loader=DataLoaderFactory(csv_file)
    >>> # data_dict creation
    >>> data_dict=data_loader.generate_data_dict()
    >>> formats=data_loader.get_supported_formats()
    >>> dataset=DatasetFactory(data_dict,{},formats,['wav'],False,0)
    >>> [first_example_id, first_tensor, first_len], = dataset[0]
    >>> print(first_example_id)
    example1
    """

    def __init__(
        self,
        data_dict,
        label_dict,
        supported_formats,
        data_entries,
        do_cache,
        cache_ram_percent,
        label_parsing_func=None,
    ):

        # Setting the variables
        self.data_dict = data_dict
        self.label_dict = label_dict
        self.supported_formats = supported_formats
        self.data_entries = data_entries
        self.do_cache = do_cache
        self.label_parsing_func = label_parsing_func

        # Creating a shared dictionary for caching
        # (dictionary must be shared across the workers)
        if do_cache:
            manager = Manager()
            self.cache = manager.dict()
            self.cache["do_caching"] = True
            self.cache_ram_percent = cache_ram_percent

    def __len__(self):
        """
        This (mandatory) function returns the length of the data list

        Returns
        -------
        int
            the number of data that can be read (i.e. length of the data_list
            entry of the data_dict)

        Example
        -------
        >>> csv_file = 'samples/audio_samples/csv_example2.csv'
        >>> # Initialization of the data_loader class
        >>> data_loader=DataLoaderFactory(csv_file)
        >>> # data_dict creation
        >>> data_dict=data_loader.generate_data_dict()
        >>> # supported formats
        >>> formats=data_loader.get_supported_formats()
        >>> # Initialization of the dataser class
        >>> dataset=DatasetFactory(data_dict,{},formats,['wav'],False,0)
        >>> # Getting data length
        >>> len(dataset)
        1
        """
        # Reading the data_list in data_dict
        data_len = len(self.data_dict["data_list"])

        return data_len

    def __getitem__(self, idx):

        # Checking if id is a tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Getting the sentence id
        snt_id = self.data_dict["data_list"][idx]

        data = []

        for data_entry in self.data_entries:

            # Reading data from data_dict
            data_line = self.data_dict[snt_id][data_entry]

            # Check if we need to convert labels to indexes
            if self.label_dict and data_entry in self.label_dict:
                lab2ind = self.label_dict[data_entry]["lab2index"]

            # Check if we need to convert labels with label func
            elif (
                self.label_parsing_func
                and "label_func" in data_line["options"]
                and (data_line["options"]["label_func"] == "True")
            ):
                lab2ind = self.label_parsing_func
            else:
                lab2ind = None
            # Managing caching

            if self.do_cache:

                if snt_id not in self.cache:

                    # Reading data
                    new_data = self.read_data(
                        data_line, snt_id, lab2ind=lab2ind
                    )
                    data.append(new_data)

                    # Store the in the variable cache if needed
                    if self.cache["do_caching"]:

                        try:
                            self.cache[snt_id] = new_data
                        except Exception:
                            pass

                        # Check ram occupation periodically
                        if random.random() < 0.05:
                            # Store data only if the RAM available is smaller
                            # than what set in cache_ram_percent.
                            if (
                                psutil.virtual_memory().percent
                                >= self.cache_ram_percent
                            ):

                                self.cache["do_caching"] = False
                else:
                    # Reading data from the cache directly
                    data.append(self.cache[snt_id])
            else:
                # Read data from the disk
                data.append(self.read_data(data_line, snt_id, lab2ind=lab2ind))

        return data

    def read_data(self, data_line, snt_id, lab2ind=None):
        """
        This function manages reading operation from disk.

        Arguments
        ---------
        data_line : dict
            One of entries extreacted from the data_dict. It contains
            all the needed information to read the data from the disk.
        snt_id : str
            Sentence identifier

        Returns
        -------
        list
            List contaning the read data. The list is formatted in the following
            way: [data_id,data_data_len]

        Example
        -------
        >>> csv_file = 'samples/audio_samples/csv_example2.csv'
        >>> # Initialization of the data_loader class
        >>> data_loader=DataLoaderFactory(csv_file)
        >>> # data_dict creation
        >>> data_dict=data_loader.generate_data_dict()
        >>> # supported formats
        >>> formats=data_loader.get_supported_formats()
        >>> # Initialization of the dataser class
        >>> dataset=DatasetFactory(data_dict,{},formats,'wav',False,0)
        >>> # data line example
        >>> data_line={'data': 'samples/audio_samples/example5.wav',
        ...           'format': 'wav',
        ...           'options': {'start': '10000', 'stop': '26000'}}
        >>> snt_id='example5'
        >>> # Reading data from disk
        >>> print(dataset.read_data(data_line, snt_id))
        ['example5', array(...)]
        """

        # Reading the data_line dictionary
        data_format = data_line["format"]
        data_source = data_line["data"]
        data_options = data_line["options"]

        # Read the data from disk
        data = self.supported_formats[data_format]["reader"](
            data_source, data_options=data_options, lab2ind=lab2ind,
        )

        # Get data_shape as float32 numpy array
        if isinstance(data, np.ndarray):
            data_shape = np.asarray(data.shape[-1]).astype("float32")
        elif isinstance(data, torch.Tensor):
            data_shape = np.asarray(data.shape[-1]).astype("float32")
        elif isinstance(data, list):
            data_shape = np.asarray(len(data)).astype("float32")
        else:
            data_shape = np.asarray(1).astype("float32")

        data_read = [snt_id, data, data_shape]

        return data_read


def convert_index_to_lab(batch, ind2lab):
    """
    Convert a batch of integer IDs to string labels

    Arguments
    ---------
    batch : list
        List of lists, a batch of sequences
    ind2lab : dict
        Mapping from integer IDs to labels

    Returns
    -------
    list
        List of lists, same size as batch, with labels from ind2lab

    Example
    -------
    >>> ind2lab = {1: "h", 2: "e", 3: "l", 4: "o"}
    >>> out = convert_index_to_lab([[4,1], [1,2,3,3,4]], ind2lab)
    >>> for seq in out:
    ...     print("".join(seq))
    oh
    hello
    """
    return [[ind2lab[int(index)] for index in seq] for seq in batch]


def relative_time_to_absolute(batch, relative_lens, rate):
    """
    Converts SpeechBrain style relative length to absolute duration

    Operates on batch level.

    Arguments
    ---------
    batch : torch.tensor
        Sequences to determine duration for.
    relative_lens : torch.tensor
        The relative length of each sequence in batch. The longest sequence in
        the batch needs to have relative length 1.0.
    rate : float
        The rate at which sequence elements occur in real world time. Sample
        rate, if batch is raw wavs (recommended) or 1/frame_shift if batch is
        features. This has to have 1/s as the unit.

    Returns
    -------
    torch.tensor
        Duration of each sequence in seconds.

    Example
    -------
    >>> batch = torch.ones(2, 16000)
    >>> relative_lens = torch.tensor([3./4., 1.0])
    >>> rate = 16000
    >>> print(relative_time_to_absolute(batch, relative_lens, rate))
    tensor([0.7500, 1.0000])
    """
    max_len = batch.shape[1]
    durations = torch.round(relative_lens * max_len) / rate
    return durations


class IterativeCSVWriter:
    """Write CSV files a line at a time.

    Arguments
    ---------
    outstream : file-object
        A writeable stream
    data_fields : list
        List of the optional keys to write. Each key will be expanded to the
        SpeechBrain format, producing three fields: key, key_format, key_opts

    Example
    -------
    >>> import io
    >>> f = io.StringIO()
    >>> writer = IterativeCSVWriter(f, ["phn"])
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    >>> writer.write("UTT1",2.5,"sil hh ee ll ll oo sil","string","")
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    UTT1,2.5,sil hh ee ll ll oo sil,string,
    >>> writer.write(ID="UTT2",phn="sil ww oo rr ll dd sil",phn_format="string")
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    UTT1,2.5,sil hh ee ll ll oo sil,string,
    UTT2,,sil ww oo rr ll dd sil,string,
    >>> writer.set_default('phn_format', 'string')
    >>> writer.write_batch(ID=["UTT3","UTT4"],phn=["ff oo oo", "bb aa rr"])
    >>> print(f.getvalue())
    ID,duration,phn,phn_format,phn_opts
    UTT1,2.5,sil hh ee ll ll oo sil,string,
    UTT2,,sil ww oo rr ll dd sil,string,
    UTT3,,ff oo oo,string,
    UTT4,,bb aa rr,string,
    """

    def __init__(self, outstream, data_fields, defaults={}):
        self._outstream = outstream
        self.fields = ["ID", "duration"] + self._expand_data_fields(data_fields)
        self.defaults = defaults
        self._outstream.write(",".join(self.fields))

    def set_default(self, field, value):
        """
        Sets a default value for the given CSV field.

        Arguments
        ---------
        field : str
            A field in the CSV
        value
            The default value
        """
        if field not in self.fields:
            raise ValueError(f"{field} is not a field in this CSV!")
        self.defaults[field] = value

    def write(self, *args, **kwargs):
        """
        Writes one data line into the CSV.

        Arguments
        ---------
        *args
            Supply every field with a value in positional form OR
        **kwargs
            Supply certain fields by key. The ID field is mandatory for all
            lines, but others can be left empty.
        """
        if args and kwargs:
            raise ValueError(
                "Use either positional fields or named fields, but not both."
            )
        if args:
            if len(args) != len(self.fields):
                raise ValueError("Need consistent fields")
            to_write = [str(arg) for arg in args]
        if kwargs:
            if "ID" not in kwargs:
                raise ValueError("I'll need to see some ID")
            full_vals = self.defaults.copy()
            full_vals.update(kwargs)
            to_write = [str(full_vals.get(field, "")) for field in self.fields]
        self._outstream.write("\n")
        self._outstream.write(",".join(to_write))

    def write_batch(self, *args, **kwargs):
        """
        Writes a batch of lines into the CSV

        Here each argument should be a list with the same length.

        Arguments
        ---------
        *args
            Supply every field with a value in positional form OR
        **kwargs
            Supply certain fields by key. The ID field is mandatory for all
            lines, but others can be left empty.
        """
        if args and kwargs:
            raise ValueError(
                "Use either positional fields or named fields, but not both."
            )
        if args:
            if len(args) != len(self.fields):
                raise ValueError("Need consistent fields")
            for arg_row in zip(*args):
                self.write(*arg_row)
        if kwargs:
            if "ID" not in kwargs:
                raise ValueError("I'll need to see some ID")
            keys = kwargs.keys()
            for value_row in zip(*kwargs.values()):
                kwarg_row = dict(zip(keys, value_row))
                self.write(**kwarg_row)

    @staticmethod
    def _expand_data_fields(data_fields):
        expanded = []
        for data_field in data_fields:
            expanded.append(data_field)
            expanded.append(data_field + "_format")
            expanded.append(data_field + "_opts")
        return expanded


# TODO: Consider making less complex
def read_wav_soundfile(file, data_options={}, lab2ind=None):  # noqa: C901
    """
    Read wav audio files with soundfile.

    Arguments
    ---------
    file : str
        The filepath to the file to read
    data_options : dict
        a dictionary containing options for the reader.
    lab2ind : dict, None
        a dictionary for converting labels to indices

    Returns
    -------
    numpy.array
        An array with the read signal

    Example
    -------
    >>> read_wav_soundfile('samples/audio_samples/example1.wav')[0:2]
    array([0.00024414, 0.00018311], dtype=float32)
    """
    # Option initialization
    start = 0
    stop = None
    endian = None
    subtype = None
    channels = None
    samplerate = None

    # List of possible options
    possible_options = [
        "start",
        "stop",
        "samplerate",
        "endian",
        "subtype",
        "channels",
    ]

    # Check if the specified options are supported
    for opt in data_options:
        if opt not in possible_options:

            err_msg = "%s is not a valid options. Valid options are %s." % (
                opt,
                possible_options,
            )

            logger.error(err_msg, exc_info=True)

    # Managing start option
    if "start" in data_options:
        try:
            start = int(data_options["start"])
        except Exception:

            err_msg = (
                "The start value for the file %s must be an integer "
                "(e.g start:405)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Managing stop option
    if "stop" in data_options:
        try:
            stop = int(data_options["stop"])
        except Exception:

            err_msg = (
                "The stop value for the file %s must be an integer "
                "(e.g stop:405)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Managing samplerate option
    if "samplerate" in data_options:
        try:
            samplerate = int(data_options["samplerate"])
        except Exception:

            err_msg = (
                "The sampling rate for the file %s must be an integer "
                "(e.g samplingrate:16000)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Managing endian option
    if "endian" in data_options:
        endian = data_options["endian"]

    # Managing subtype option
    if "subtype" in data_options:
        subtype = data_options["subtype"]

    # Managing channels option
    if "channels" in data_options:
        try:
            channels = int(data_options["channels"])
        except Exception:

            err_msg = (
                "The number of channels for the file %s must be an integer "
                "(e.g channels:2)" % (file)
            )

            logger.error(err_msg, exc_info=True)

    # Reading the file with the soundfile reader
    try:
        [signal, fs] = sf.read(
            file,
            start=start,
            stop=stop,
            samplerate=samplerate,
            endian=endian,
            subtype=subtype,
            channels=channels,
        )

        signal = signal.astype("float32")

    except RuntimeError as e:
        err_msg = "cannot read the wav file %s" % (file)
        e.args = (*e.args, err_msg)
        raise

    # Set time_steps always last as last dimension
    if len(signal.shape) > 1:
        signal = signal.transpose()

    return signal


def read_pkl(file, data_options={}, lab2ind=None):
    """
    This function reads tensors store in pkl format.

    Arguments
    ---------
    file : str
        The path to file to read.
    data_options : dict, optional
        A dictionary containing options for the reader.
    lab2ind : dict, optional
        Mapping from label to integer indices.

    Returns
    -------
    numpy.array
        The array containing the read signal
    """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            pkl_element = pickle.load(f)
    except pickle.UnpicklingError:
        err_msg = "cannot read the pkl file %s" % (file)
        raise ValueError(err_msg)

    type_ok = False

    if isinstance(pkl_element, list):

        if isinstance(pkl_element[0], float):
            tensor = torch.FloatTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], int):
            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], str):

            # convert string to integer as specified in self.label_dict
            if lab2ind is not None:
                for index, val in enumerate(pkl_element):
                    pkl_element[index] = lab2ind[val]

            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if not (type_ok):
            err_msg = (
                "The pkl file %s can only contain list of integers, "
                "floats, or strings. Got %s"
            ) % (file, type(pkl_element[0]))
            raise ValueError(err_msg)
    else:
        tensor = pkl_element

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int32")

    return tensor


def read_string(string, data_options={}, lab2ind=None):
    """
    This function reads data in string format.

    Arguments
    ---------
    string : str
        String to read
    data_options : dict, optional
        Options for the reader
    lab2ind : dict, optional
        Mapping from label to index

    Returns
    -------
    torch.LongTensor
        The read string in integer indices, if lab2ind is provided, else
    list
        The read string split at each space

    Example
    -------
    >>> read_string('hello world', lab2ind = {"hello":1, "world": 2})
    tensor([1, 2])
    """

    if callable(lab2ind):
        return lab2ind(string)

    # Splitting elements with ' '
    string = string.split(" ")

    # convert string to integer as specified in self.label_dict
    if lab2ind is not None:
        for index, val in enumerate(string):
            if val not in lab2ind:
                lab2ind[val] = len(lab2ind)

            string[index] = lab2ind[val]

        string = torch.LongTensor(string)

    return string


def read_kaldi_lab(kaldi_ali, kaldi_lab_opts):
    """
    Read labels in kaldi format

    Uses kaldi IO

    Arguments
    ---------
    kaldi_ali : str
        Path to directory where kaldi alignents are stored.
    kaldi_lab_opts : str
        A string that contains the options for reading the kaldi alignments.

    Returns
    -------
    dict
        A dictionary contaning the labels

    Note
    ----
    This depends on kaldi-io-for-python. Install it separately.
    See: https://github.com/vesis84/kaldi-io-for-python

    Example
    -------
    This example requires kaldi files
    ```
    lab_folder = '/home/kaldi/egs/TIMIT/s5/exp/dnn4_pretrain-dbn_dnn_ali'
    read_kaldi_lab(lab_folder, 'ali-to-pdf')
    ```
    """
    # EXTRA TOOLS
    try:
        import kaldi_io
    except ImportError:
        raise ImportError("Could not import kaldi_io. Install it to use this.")
    # Reading the Kaldi labels
    lab = {
        k: v
        for k, v in kaldi_io.read_vec_int_ark(
            "gunzip -c "
            + kaldi_ali
            + "/ali*.gz | "
            + kaldi_lab_opts
            + " "
            + kaldi_ali
            + "/final.mdl ark:- ark:-|",
        )
    }
    return lab


def write_wav_soundfile(data, filename, sampling_rate):
    """
    Can be used to write audio with soundfile

    Expecting data in (time, [channels]) format

    Arguments
    ---------
    data : torch.tensor
        it is the tensor to store as and audio file
    filename : str
        path to file where writing the data
    sampling_rate : int, None
        sampling rate of the audio file

    Returns
    -------
    None

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal = 0.1*torch.rand([16000])
    >>> tmpfile = os.path.join(tmpdir, 'wav_example.wav')
    >>> write_wav_soundfile(signal, tmpfile, sampling_rate=16000)
    """
    if len(data.shape) > 2:
        err_msg = (
            "expected signal in the format (time, channel). Got %s "
            "dimensions instead of two for file %s"
            % (len(data.shape), filename)
        )
        raise ValueError(err_msg)
    if isinstance(data, torch.Tensor):
        if len(data.shape) == 2:
            data = data.transpose(0, 1)
        # Switching to cpu and converting to numpy
        data = data.cpu().numpy()
    # Writing the file
    sf.write(filename, data, sampling_rate)


def write_txt_file(data, filename, sampling_rate=None):
    """
    Write data in text format

    Arguments
    ---------
    data : str, list, torch.tensor, numpy.ndarray
        The data to write in the text file
    filename : str
        Path to file where to write the data
    sampling_rate : None
        Not used, just here for interface compatibility

    Returns
    -------
    None

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal=torch.tensor([1,2,3,4])
    >>> write_txt_file(signal, os.path.join(tmpdir, 'example.txt'))
    """
    del sampling_rate  # Not used.
    # Check if the path of filename exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, list):
            for line in data:
                print(line, file=fout)
        if isinstance(data, str):
            print(data, file=fout)


def write_stdout(data, filename=None, sampling_rate=None):
    """
    Write data to standard output

    Arguments
    ---------
    data : str, list, torch.tensor, numpy.ndarray
        The data to write in the text file
    filename : None
        Not used, just here for compatibility
    sampling_rate : None
        Not used, just here for compatibility

    Returns
    -------
    None


    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal = torch.tensor([[1,2,3,4]])
    >>> write_stdout(signal, tmpdir + '/example.txt')
    [1, 2, 3, 4]
    """
    # Managing Torch.Tensor
    if isinstance(data, torch.Tensor):
        data = data.tolist()
    # Managing np.ndarray
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        for line in data:
            print(line)
    if isinstance(data, str):
        print(data)


def save_img(data, filename, sampling_rate=None, logger=None):
    """
    Save a tensor as an image.

    Arguments
    ---------
    data : torch.tensor
        The tensor to write in the text file
    filename : str
        Path where to write the data.
    sampling_rate : int
        Sampling rate of the audio file.

    Returns
    -------
    None

    Note
    ----
    Depends on matplotlib as an extra tool. Install it separately.

    Example
    -------
    >>> tmpdir = getfixture('tmpdir')
    >>> signal=torch.rand([100,200])
    >>> try:
    ...     save_img(signal, tmpdir + '/example.png')
    ... except ImportError:
    ...     pass
    """
    # EXTRA TOOLS
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        err_msg = "Cannot import matplotlib. To use this, install matplotlib"
        raise ImportError(err_msg)
    # Checking tensor dimensionality
    if len(data.shape) < 2 or len(data.shape) > 3:
        err_msg = (
            "cannot save image  %s. Save in png format supports 2-D or "
            "3-D tensors only (e.g. [x,y] or [channel,x,y]). "
            "Got %s" % (filename, str(data.shape))
        )
        raise ValueError(err_msg)
    if len(data.shape) == 2:
        N_ch = 1
    else:
        N_ch = data.shape[0]
    # Flipping axis
    data = data.flip([-2])
    for i in range(N_ch):
        if N_ch > 1:
            filename = filename.replace(".png", "_ch_" + str(i) + ".png")
        if N_ch > 1:
            plt.imsave(filename, data[i])
        else:
            plt.imsave(filename, data)


class TensorSaver(torch.nn.Module):
    """
    Save tensors on disk.

    Arguments
    ---------
    save_folder : str
        The folder where the tensors are stored.
    save_format : str, optional
        Default: "pkl"
        The format to use to save the tensor.
        See get_supported_formats() for an overview of
        the supported data formats.
    save_csv : bool, optional
        Default: False
        If True it saves the list of data written in a csv file.
    data_name : str, optional
        Default: "data"
        The name to give to saved data
    parallel_write : bool, optional
        Default: False
        If True it saves the data using parallel processes.
    transpose : bool, optional
        Default: False
        if True it transposes the data matrix
    decibel : bool, optional
        Default: False
        if True it saves the log of the data.

    Example:
    >>> tmpdir = getfixture('tmpdir')
    >>> save_signal = TensorSaver(save_folder=tmpdir, save_format='wav')
    >>> signal = 0.1 * torch.rand([1, 16000])
    >>> save_signal(signal, ['example_random'], torch.ones(1))
    """

    def __init__(
        self,
        save_folder,
        save_format="pkl",
        save_csv=False,
        data_name="data",
        sampling_rate=16000,
        parallel_write=False,
        transpose=False,
        decibel=False,
    ):
        super().__init__()

        self.save_folder = save_folder
        self.save_format = save_format
        self.save_csv = save_csv
        self.data_name = data_name
        self.sampling_rate = sampling_rate
        self.parallel_write = parallel_write
        self.transpose = transpose
        self.decibel = decibel

        # Definition of other variables
        self.supported_formats = self.get_supported_formats()

        # Creating the save folder if it does not exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Check specified format
        if self.save_format not in self.supported_formats:
            err_msg = (
                "the format %s specified in the config file is not "
                "supported. The current version supports %s"
                % (self.save_format, self.supported_formats.keys())
            )
            logger.error(err_msg)

        # Create the csv file (if specified)
        if self.save_csv:
            self.save_csv_path = os.path.join(self.save_folder, "csv.csv")
            open(self.save_csv_path, "w").close()
            self.first_line_csv = True

    def forward(self, data, data_id, data_len):
        """
        Arguments
        ---------
        data : torch.tensor
            batch of audio signals to save
        data_id : list
            list of ids in the batch
        data_len : torch.tensor
            length of each audio signal
        """
        # Convertion to log (if specified)
        if self.decibel:
            data = 10 * data.log10()

        # Writing data on disk (in parallel)
        self.write_batch(data, data_id, data_len)

    def write_batch(self, data, data_id, data_len):
        """
        Saves a batch of data.

        Arguments
        ---------
        data : torch.tensor
            batch of audio signals to save
        data_id : list
            list of ids in the batch
        data_len : torch.tensor
            relative length of each audio signal

        Example
        -------
        >>> save_folder = getfixture('tmpdir')
        >>> save_format = 'wav'
        >>> save_signal=TensorSaver(save_folder, save_format)
        >>> # random signal
        >>> signal=0.1*torch.rand([1,16000])
        >>> # saving
        >>> save_signal.write_batch(signal, ['example_random'], torch.ones(1))
        """

        # Write in parallel all the examples in the batch on disk:
        jobs = []

        # Move time dimension last
        data = data.transpose(1, -1)

        # Multiprocessing on gpu is something we have to fix
        data = data.cpu()

        if self.save_csv:
            csv_f = open(self.save_csv_path, "a")

            if self.first_line_csv:
                line = "ID, duration, %s, %s_format, %s_opts\n" % (
                    self.data_name,
                    self.data_name,
                    self.data_name,
                )
                self.first_line_csv = False

                csv_f.write(line)

        # Processing all the batches in data
        for j in range(data.shape[0]):

            # Selection up to the true data length (without padding)
            actual_size = int(torch.round(data_len[j] * data[j].shape[0]))
            data_save = data[j].narrow(0, 0, actual_size)

            # Transposing the data if needed
            if self.transpose:
                data_save = data_save.transpose(-1, -2)

            # Selection of the needed data writer
            writer = self.supported_formats[self.save_format]["writer"]

            # Output file
            data_file = os.path.join(
                self.save_folder, data_id[j] + "." + self.save_format
            )

            # Writing all the batches in parallel (if paralle_write=True)
            if self.parallel_write:
                p = mp.Process(
                    target=writer,
                    args=(data_save, data_file),
                    kwargs={"sampling_rate": self.sampling_rate},
                )
                p.start()
                jobs.append(p)
            else:
                # Writing data on disk with the selected writer
                writer(
                    data_save, data_file, sampling_rate=self.sampling_rate,
                )

            # Saving csv file
            if self.save_csv:
                line = "%s, %f, %s, %s, ,\n" % (
                    data_id[j],
                    actual_size,  # We are here saving the number of time steps
                    data_file,
                    self.save_format,
                )
                csv_f.write(line)

        # Waiting all jobs to finish
        if self.parallel_write:
            for j in jobs:
                j.join()

        # Closing the csv file
        if self.save_csv:
            csv_f.close()

    @staticmethod
    def get_supported_formats():
        """
        Lists the supported formats and their related writers

        Returns
        -------
        dict
            Maps from file name extensions to dicts which have the keys
            "writer" and "description"

        Example
        -------
        >>> save_folder = getfixture('tmpdir')
        >>> save_format = 'wav'
        >>> # class initialization
        >>> saver = TensorSaver(save_folder, save_format)
        >>> saver.get_supported_formats()['wav']
        {'writer': <function ...>, 'description': ...}
        """

        # Dictionary initialization
        supported_formats = {}

        # Adding sound file supported formats
        sf_formats = sf.available_formats()

        for wav_format in sf_formats.keys():
            wav_format = wav_format.lower()
            supported_formats[wav_format] = {}
            supported_formats[wav_format]["writer"] = write_wav_soundfile
            supported_formats[wav_format]["description"] = sf_formats[
                wav_format.upper()
            ]

        # Adding the other supported formats
        supported_formats["pkl"] = {}
        supported_formats["pkl"]["writer"] = save_pkl
        supported_formats["pkl"]["description"] = "Python binary format"

        supported_formats["txt"] = {}
        supported_formats["txt"]["writer"] = write_txt_file
        supported_formats["txt"]["description"] = "Plain text"

        supported_formats["png"] = {}
        supported_formats["png"]["writer"] = save_img
        supported_formats["png"]["description"] = "image in png format"

        supported_formats["stdout"] = {}
        supported_formats["stdout"]["writer"] = write_stdout
        supported_formats["stdout"]["description"] = "write on stdout"

        return supported_formats


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """
    Creates a binary mask for each sequence.
    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : The binary mask

    Example:
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def get_md5(file):
    """
    Get the md5 checksum of an input file

    Arguments
    ---------
    file : str
        Path to file for which compute the checksum

    Returns
    -------
    md5
        Checksum for the given filepath

    Example
    -------
    >>> get_md5('samples/audio_samples/example1.wav')
    'c482d0081ca35302d30d12f1136c34e5'
    """
    # Lets read stuff in 64kb chunks!
    BUF_SIZE = 65536
    md5 = hashlib.md5()
    # Computing md5
    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def save_md5(files, out_file):
    """
    Saves the md5 of a list of input files as a pickled dict into a file.

    Arguments
    ---------
    files : list
        List of input files from which we will compute the md5.
    outfile : str
        The path where to store the output pkl file.

    Returns
    -------
    None

    Example:
    >>> files = ['samples/audio_samples/example1.wav']
    >>> tmpdir = getfixture('tmpdir')
    >>> save_md5(files, os.path.join(tmpdir, "md5.pkl"))
    """
    # Initialization of the dictionary
    md5_dict = {}
    # Computing md5 for all the files in the list
    for file in files:
        md5_dict[file] = get_md5(file)
    # Saving dictionary in pkl format
    save_pkl(md5_dict, out_file)


def save_pkl(obj, file):
    """
    Save an object in pkl format.

    Arguments
    ---------
    obj : object
        Object to save in pkl format
    file : str
        Path to the output file
    sampling_rate : int
        Sampling rate of the audio file, TODO: this is not used?

    Example:
    >>> tmpfile = os.path.join(getfixture('tmpdir'), "example.pkl")
    >>> save_pkl([1, 2, 3, 4, 5], tmpfile)
    >>> load_pkl(tmpfile)
    [1, 2, 3, 4, 5]
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    """
    Loads a pkl file

    For an example, see `save_pkl`

    Arguments
    ---------
    file : str
        Path to the input pkl file.

    Returns
    -------
    The loaded object
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def prepend_bos_token(label, bos_index):
    """Create labels with <bos> token at the beginning.

    Arguments
    ---------
    label : torch.IntTensor
        Containing the original labels. Must be of size: [batch_size, max_length]
    bos_index : int
        The index for <bos> token.

    Returns
    -------
    new_label : The new label with <bos> at the beginning.

    Example:
    >>> label=torch.LongTensor([[1,0,0], [2,3,0], [4,5,6]])
    >>> new_label=prepend_bos_token(label, bos_index=7)
    >>> new_label
    tensor([[7, 1, 0, 0],
            [7, 2, 3, 0],
            [7, 4, 5, 6]])
    """
    new_label = label.long().clone()
    batch_size = label.shape[0]

    bos = new_label.new_zeros(batch_size, 1).fill_(bos_index)
    new_label = torch.cat([bos, new_label], dim=1)
    return new_label


def append_eos_token(label, length, eos_index):
    """Create labels with <eos> token appended.

    Arguments
    ---------
    label : torch.IntTensor
        Containing the original labels. Must be of size: [batch_size, max_length]
    length : torch.LongTensor
        Cotaining the original length of each label sequences. Must be 1D.
    eos_index : int
        The index for <eos> token.

    Returns
    -------
    new_label : The new label with <eos> appended.

    Example:
    >>> label=torch.IntTensor([[1,0,0], [2,3,0], [4,5,6]])
    >>> length=torch.LongTensor([1,2,3])
    >>> new_label=append_eos_token(label, length, eos_index=7)
    >>> new_label
    tensor([[1, 7, 0, 0],
            [2, 3, 7, 0],
            [4, 5, 6, 7]], dtype=torch.int32)
    """
    new_label = label.int().clone()
    batch_size = label.shape[0]

    pad = new_label.new_zeros(batch_size, 1)
    new_label = torch.cat([new_label, pad], dim=1)
    new_label[torch.arange(batch_size), length.long()] = eos_index
    return new_label


def merge_char(sequences, space="_"):
    """Merge characters sequences into word sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains character sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contain word sequences for each sentence.

    Example:
    >>> sequences = [["a", "b", "_", "c", "_", "d", "e"], ["e", "f", "g", "_", "h", "i"]]
    >>> results = merge_char(sequences)
    >>> results
    [['ab', 'c', 'de'], ['efg', 'hi']]
    """
    results = []
    for seq in sequences:
        words = "".join(seq).split("_")
        results.append(words)
    return results


def merge_csvs(data_folder, csv_lst, merged_csv):
    """Merging several csv files into one file.

    Arguments
    ---------
    data_folder : string
        The folder to store csv files to be merged and after merging.
    csv_lst : list
        filenames of csv file to be merged.
    merged_csv : string
        The filename to write the merged csv file.


    Example:
    >>> merge_csvs("samples/audio_samples/",
    ... ["csv_example.csv", "csv_example2.csv"],
    ... "test_csv_merge.csv")
    """
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        logger.info("Skipping merging. Completed in previous run.")

    with open(os.path.join(data_folder, csv_lst[0])) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_lst:
        with open(os.path.join(data_folder, csv_file)) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError(
                            "Different header for " f"{csv_lst[0]} and {csv}."
                        )
                    continue
                lines.append(line)
    with open(write_path, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    logger.info(f"{write_path} is created.")


def split_word(sequences, space="_"):
    """Split word sequences into character sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains words sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contain word sequences for each sentence.

    Example:
    >>> sequences = [['ab', 'c', 'de'], ['efg', 'hi']]
    >>> results = split_word(sequences)
    >>> results
    [['a', 'b', '_', 'c', '_', 'd', 'e'], ['e', 'f', 'g', '_', 'h', 'i']]
    """
    results = []
    for seq in sequences:
        chars = list("_".join(seq))
        results.append(chars)
    return results
