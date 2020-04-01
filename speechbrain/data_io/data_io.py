"""
-----------------------------------------------------------------------------
 data_io.py

 Description: This library gathers that implement support functions for
              data i/o operations.
 -----------------------------------------------------------------------------
"""

import io
import os
import re
import csv
import sys
import gzip
import torch
import struct
import psutil
import random
import pickle
import logging
import hashlib
import threading
import subprocess
import numpy as np
import soundfile as sf
import multiprocessing as mp
from itertools import groupby
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
from speechbrain.utils.data_utils import recursive_items
logger = logging.getLogger(__name__)


class create_dataloader(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     data_io.create_dataloader (author: Mirco Ravanelli)

     Description:
        This class creates the data_loaders for the given csv file.

     Args:
        csv: it is the csv file that itemized the data.

        - batch_size: (type: int(1, inf), default: 1):
           the data itemized in the csv file are automatically
           organized in batches. In the case of variable size
           tensors, zero padding is performed. When batch_size=1,
           the data are simply processed one by one without the
           creation of batches.
        csv_read: this option can be used to read only some data_entries of
           the csv file. When not specified, it automatically reads
           all the data entries.

       - sentence_sorting: ('ascending,descending,random,original',
                default: 'original'):
           This parameter specifies how to sort the data
           before the batch creation. Ascending and
           descending values sort the data using the
           "duration" field in the csv files. Random sort
           the data randomly, while original (the default
           option) keeps the original sequence of data
           defined in the csv file. Note that this option
           affects the batch creation. If the data are
           sorted in ascending or descending order the
           batches will approximately have the same size
           and the need for zero padding is minimized.
           Instead, if sentence_sorting is set to random,
           the batches might be composed of both short and
           long sequences and several zeros might be added
           in the batch. When possible, it is desirable to
           sort the data. This way, we use more
           efficiently the computational resources,
           without wasting time on processing time steps
           composed on zeros only. Note that is the data
           are sorted in ascending/ descending errors the
           same batches will be created every time we
           want to loop over the dataset, while if we set
           a random order the batches will be different
           every time we loop over the dataset.
        select_n_sentences: this option can be used to read-only n
           sentences from the csv file. This option can be
           useful to debug the code, when instead of
           running an experiment of a full set of data I
           might just want to run it with a little about
           of data.
        num_workers: data are read using the pytorch data_loader.
           This option set the number of workers used to
           read the data from disk and form the related
           batch of data. Please, see the pytorch
           documentation on the data loader for more
           details.
        cache: When set to true, this option stores the input
           data in a variable called self.cache (see
           create_dataloader in data_io.py). In practice,
           the first time the data are read from the disk,
           they are stored in the cpu RAM. If the data
           needs to be used again (e.g. when loops>1)
           the data will be read from the RAM directly.
           If False, data are read from the disk every
           time.  Data are stored until a certain
           percentage of the total ram available is
           reached (see cache_ram_percent below).
        cache_ram_percent: If cache if True, data will be stored in the
           cpu RAM until the total RAM occupation is less
           or equal than the specified threshold
           (by default 75%). In practice, if a lot of RAM
           is available several data will be stored in
           memory, otherwise, most of them will be read
           from the disk directly.
        drop_last: this is an option directly passed to the
           pytorch dataloader (see the related
           documentation for more details). When True,
           it skips the last batch of data if contains
           fewer samples than the other ones.
        replacements: String replacements to perform in this method
        output_folder: A folder for storing the label dict


     Example:   from speechbrain.data_io.data_io import create_dataloader

                config={'class_name':'core.loop',\
                         'csv_file':'samples/audio_samples/csv_example2.csv'}

                # Initialization of the class
                data_loader=create_dataloader(config)

                print(data_loader([]))
     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        csv_file,
        batch_size=1,
        csv_read=None,
        sentence_sorting='original',
        num_workers=0,
        cache=False,
        cache_ram_percent=75,
        select_n_sentences=None,
        drop_last=False,
        padding_value=0,
        replacements={},
        output_folder=None,
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
        self.drop_last = drop_last
        self.padding_value = padding_value
        self.replacements = replacements
        self.output_folder = output_folder

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

        if self.output_folder:
            self.label_dict = self.label_dict_creation(data_dict)
        else:
            self.label_dict = None

        self.data_len = len(data_dict["data_list"])

        if self.csv_read is None:
            self.csv_read = data_dict["data_entries"]

        self.dataloader = []

        # Creating a dataloader for each data entry in the csv file
        for data_entry in self.csv_read:

            dataset = create_dataset(
                data_dict,
                self.label_dict,
                self.supported_formats,
                data_entry,
                self.cache,
                self.cache_ram_percent,
            )

            self.dataloader.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    pin_memory=False,
                    drop_last=self.drop_last,
                    num_workers=self.num_workers,
                    collate_fn=self.batch_creation,
                )
            )

        return self.dataloader

    def batch_creation(self, data_list):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.batch_creation (author: Mirco Ravanelli)

         Description: This function create the batch of data. When necessary
                      it performs zero padding on the input tensors. The
                      function is executed in collate_fn of the pytorch
                      DataLoader.

         Args:
        self: - data_list (type: list, mandatory):
                        it is the list of data returned by the data reader
                        [data_id,data,data_len]

         Output:
        batch: it is a list containing the final batches:
                        [data_id,data,data_len] where zero-padding is
                        performed where needed.

         Example:   from speechbrain.data_io.data_io import create_dataloader

                    config={'class_name':'core.loop',\
                             'csv_file':'samples/audio_samples/\
                                 csv_example2.csv'}

                    # Initialization of the class
                    data_loader=create_dataloader(config)

                    print(data_loader([]))
         --------------------------------------------.------------------------
         """

        # The data_list is always compose by [id,data,data_len]
        data_list = list(map(list, zip(*data_list)))

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
            time_steps = time_steps / batch_data.shape[-1]

        else:
            # Non-tensor case
            batch_data = sequences

        # Batch composition
        batch = [snt_ids, batch_data, time_steps]

        return batch

    def padding(self, sequences):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.padding (author: Mirco Ravanelli)

         Description: This function perform zero padding on the input list of
                      tensors.

         Args:
        self: - sequences (type: list, mandatory):
                        it is the list of tensor to pad.

         Output:
        batch_data: it is a tensor gathering all the padded tensors.

         Example:   import torch
                    from speechbrain.data_io.data_io import create_dataloader

                    config={'class_name':'core.loop',\
                             'csv_file':'samples/audio_samples/\
                                 csv_example2.csv'}

                    # Initialization of the class
                    data_loader=create_dataloader(config)

                   # list of tensors
                   tensor_lst=[torch.tensor([1,2,3,4]),torch.tensor([1,2])]

                   # Applying zero padding
                   print(data_loader.padding(tensor_lst))
         --------------------------------------------.------------------------
         """

        # Batch size
        batch_size = len(sequences)

        # Computing data dimensionality (only the first time)
        try:
            self.data_dim
        except Exception:
            self.data_dim = list(sequences[0].shape[:-1])

        # Finding the max len across sequences
        max_len = max([s.size(-1) for s in sequences])

        # Batch out dimensions
        out_dims = [batch_size] + self.data_dim + [max_len]

        # Batch initialization
        batch_data = torch.zeros(out_dims) + self.padding_value

        # Appending data
        for i, tensor in enumerate(sequences):
            length = tensor.shape[-1]
            batch_data[i, ..., :length] = tensor

        return batch_data

    def numpy2torch(self, data_list):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.numpy2torch (author: Mirco Ravanelli)

         Description: This function coverts a list of numpy tensors to
                       torch.Tensor

         Args:
        self: - data_list (type: list, mandatory):
                        it is a list of numpy arrays.

         Output:    None

         Example:  import numpy as np
                   from speechbrain.data_io.data_io import create_dataloader

                   config={'class_name':'core.loop',\
                             'csv_file':'samples/audio_samples/\
                                 csv_example2.csv'}

                   # Initialization of the class
                   data_loader=create_dataloader(config)

                   # list of numpy tensors
                   tensor_lst=[[np.asarray([1,2,3,4]),np.asarray([1,2])]]

                   # Applying zero padding
                   data_loader.numpy2torch(tensor_lst)
                   print(tensor_lst)
                   print(type(tensor_lst[0][0]))
         ---------------------------------------------------------------------
         """

        # Covert all the elements of the list to torch.Tensor
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                if isinstance(data_list[i][j], np.ndarray):
                    data_list[i][j] = torch.from_numpy(data_list[i][j])

    def label_dict_creation(self, data_dict):

        label_dict_file = (
            self.output_folder + "/label_dict.pkl"
        )

        # Read previously stored label_dict
        if os.path.isfile(label_dict_file):
            label_dict = load_pkl(label_dict_file)
        else:
            # create label counts and label2index automatically when needed
            label_dict = {}

        # Update label dict
        for snt in data_dict:
            if isinstance(data_dict[snt], dict):
                for elem in data_dict[snt]:

                    if "format" in data_dict[snt][elem]:

                        count_lab = False
                        opts = data_dict[snt][elem]["options"]

                        if data_dict[snt][elem]["format"] == "string" and (
                            "label" not in opts or opts["label"] == "True"
                        ):

                            if (
                                len(
                                    data_dict[snt][elem]["data"].split(" ")
                                )
                                > 1
                            ):
                                # Processing list of string labels
                                labels = data_dict[snt][elem][
                                    "data"
                                ].split(" ")
                                count_lab = True

                            else:
                                # Processing a single label
                                labels = [data_dict[snt][elem]["data"]]
                                count_lab = True

                        if data_dict[snt][elem]["format"] == "pkl":

                            labels = load_pkl(data_dict[snt][elem]["data"])

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

                        if count_lab:
                            if elem not in label_dict:
                                label_dict[elem] = {}
                                label_dict[elem]["counts"] = {}

                            for lab in labels:
                                if lab not in label_dict[elem]["counts"]:
                                    label_dict[elem]["counts"][lab] = 1
                                else:
                                    label_dict[elem]["counts"][lab] = (
                                        label_dict[elem]["counts"][lab] + 1
                                    )

        # create label2index:
        for lab in label_dict:
            # sorted_ids = sorted(label_dict[lab]["counts"].keys())
            cnt_id = 0

            label_dict[lab]["lab2index"] = {}
            label_dict[lab]["index2lab"] = {}
            for lab_id in label_dict[lab]["counts"]:
                label_dict[lab]["lab2index"][lab_id] = cnt_id
                label_dict[lab]["index2lab"][cnt_id] = lab_id
                cnt_id = cnt_id + 1

        # saving the label_dict:
        save_pkl(label_dict, label_dict_file)

        return label_dict

    def generate_data_dict(self,):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.generate_data_dict(author: Mirco Ravanelli)

         Description: This function creates a dictionary from the csv file

         Args:
        self: Output:
        data_dict: it is a dictionary with the data itemized in the csv
                       file.

         Example:  from speechbrain.data_io.data_io import create_dataloader

                   config={'class_name':'core.loop',\
                             'csv_file':'samples/audio_samples/\
                                 csv_example2.csv'}

                   # Initialization of the class
                   data_loader=create_dataloader(config)

                   print(data_loader.generate_data_dict())
         --------------------------------------------.------------------------
         """

        # Initial prints
        msg = "\tCreating dataloader for %s" % (self.csv_file)
        logger.debug(msg)

        # Initialization of the data_dict
        data_dict = {}

        # CSV file reader
        reader = csv.reader(open(self.csv_file, "r"))

        first_row = True

        # Tracking the total sentence duration
        total_duration = 0

        for row in reader:

            # Skipping empty lines
            if len(row) == 0:
                continue

            # remove left/right spaces
            row = [elem.strip(" ") for elem in row]

            if first_row:

                # Make sure ID field exists
                if "ID" not in row:
                    err_msg = (
                        "The mandatory field ID (i.e, the field that contains "
                        "a unique  id for each sentence is not present in the "
                        "csv file  %s" % (self.csv_file)
                    )

                    logger.error(err_msg, exc_info=True)

                # Make sure the duration field exists
                if "duration" not in row:
                    err_msg = (
                        "The mandatory field duration (i.e, the field that "
                        "contains the  duration of each sentence is not "
                        "present in the csv  file %s" % (self.csv_file)
                    )

                    logger.error(err_msg, exc_info=True)

                if len(row) == 2:
                    err_msg = (
                        "The cvs file %s does not contain features entries! "
                        "The features are specified with the following fields:"
                        "feaname, feaname_format, feaname_opts"
                        % (self.csv_file)
                    )

                    logger.error(err_msg, exc_info=True)

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

                        logger.error(err_msg, exc_info=True)

                    if feat_name + "_opts" not in row:
                        err_msg = (
                            "The feature %s in the cvs file %s does not "
                            "contain the field %s to specified the reader "
                            "options. "
                            % (feat_name, self.csv_file, feat_name + "_opts")
                        )

                        logger.error(err_msg, exc_info=True)

                # Store the field list
                field_lst = row

                first_row = False

            else:

                # replace local variables with global ones
                variable_finder = re.compile(r'\$[\w.]+')
                for i, item in enumerate(row):
                    try:
                        variable_finder.sub(
                            lambda x: self.replacements[x[0]],
                            item,
                        )
                    except KeyError:
                        log_msg = ("The item '%s' contains variables "
                                   "not included in 'replacements'" % item)
                        logger.error(log_msg)

                # Make sure that the current row contains all the fields
                if len(row) != len(field_lst):
                    err_msg = (
                        'The row "%s" of the cvs file %s does not '
                        "contain the right number fields (they must be %i "
                        "%s"
                        ")" % (row, self.csv_file, len(field_lst), field_lst)
                    )

                    logger.error(err_msg, exc_info=True)

                # Filling the data dictionary
                for i, field in enumerate(field_lst):

                    field_name = row[i]

                    if i == 0:
                        id_field = field_name
                        data_dict[id_field] = {}
                    else:

                        if field == "duration":
                            data_dict[id_field][field] = {}
                            duration = float(row[i])
                            data_dict[id_field][field] = row[i]
                            total_duration = total_duration + duration
                        else:

                            field_or = field
                            field = field.replace("_format", "").replace(
                                "_opts", ""
                            )

                            if field not in data_dict[id_field]:
                                data_dict[id_field][field] = {}
                                data_dict[id_field][field]["data"] = {}
                                data_dict[id_field][field]["format"] = {}
                                data_dict[id_field][field]["options"] = {}

                            if "_format" in field_or:
                                data_dict[id_field][field]["format"] = row[i]

                            elif "_opts" in field_or:
                                data_dict[id_field][field]["options"] = {}
                                if len(row[i]) > 0:
                                    lst_opt = row[i].split(" ")
                                    for opt in lst_opt:
                                        opt_name = opt.split(":")[0]
                                        opt_val = opt.split(":")[1]

                                        data_dict[id_field][field]["options"][
                                            opt_name
                                        ] = {}
                                        data_dict[id_field][field]["options"][
                                            opt_name
                                        ] = opt_val

                            else:
                                data_dict[id_field][field]["data"] = row[i]

        data_dict = self.sort_sentences(data_dict, self.sentence_sorting)

        log_text = (
            "\tNumber of sentences: %i\n" % (len(data_dict.keys()))
            + "\tTotal duration (hours): %1.2f \n" % (total_duration / 3600)
            + "\tAverage duration (seconds): %1.2f \n"
            % (total_duration / len(data_dict.keys()))
        )

        logger.debug(log_text)

        # Adding sorted list of sentences
        data_dict["data_list"] = list(data_dict.keys())

        snt_ex = data_dict["data_list"][0]

        data_entries = list(data_dict[snt_ex].keys())
        data_entries.remove("duration")

        data_dict["data_entries"] = data_entries

        return data_dict

    @staticmethod
    def sort_sentences(data_dict, sorting):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.sort_sentences (author: Mirco Ravanelli)

         Description: This function sorts the data dictionary as specified.

         Args:
        self: - data_dict (type: dict, mandatory):
                       it is a dictionary with the data itemized in the csv
                       file.
        sorting: mandatory):
                           it is a dictionary with the data itemized in the
                           csv file.

         Output:
        sorted_dictionary: it is a dictionary with the sorted data

         Example:  from speechbrain.data_io.data_io import create_dataloader

                   config={'class_name':'core.loop',\
                             'csv_file':'samples/audio_samples/\
                                 csv_example2.csv'}

                   # Initialization of the class
                   data_loader=create_dataloader(config)

                   # data_dict creation
                   data_dict=data_loader.generate_data_dict()
                   del data_dict['data_list']
                   del data_dict['data_entries']

                   print(data_loader.sort_sentences(data_dict,'original'))
                   print(data_loader.sort_sentences(data_dict,'random'))
                   print(data_loader.sort_sentences(data_dict,'ascending'))
         --------------------------------------------.------------------------
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

        # Random sorting
        if sorting == "random":
            sorted_ids = list(data_dict.keys())
            random.shuffle(sorted_ids)

        # Original order
        if sorting == "original":
            sorted_ids = list(data_dict.keys())

        # Filling the dictionary
        for snt_id in sorted_ids:
            sorted_dictionary[snt_id] = data_dict[snt_id]

        return sorted_dictionary

    @staticmethod
    def get_supported_formats():
        """
        ---------------------------------------------------------------------
        data_io.create_dataloader.get_supported_formats
        (author: Mirco Ravanelli)

        Description: This function itemize the supported reading formats

        Returns:
            a dictionary contained the supported formats and the related
            readers.

        Example:
            >>> data_loader=create_dataloader(
            ...     csv_file='samples/audio_samples/csv_example2.csv'
            ... )
            >>> data_loader.get_supported_formats()

         --------------------------------------------.------------------------
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


class create_dataset(Dataset):
    """
     -------------------------------------------------------------------------
     data_io.create_dataset (author: Mirco Ravanelli)

     Description: This class (of type dataset) creates the dataset needed by
                  the pytorch data_loader.

     Input (init):  - data_dict, (type: dict, mandatory):
                       it is a dictionary containing all the entries of the
                       csv file.
        supported_formats: it is a dictionary contaning the reading supported
                       format.
        data_entry: it is a list containing the data_entries to read from
                       the csv file.

                    - do_cache(bool,Default:False):
                       When set to true, this option stores the input data
                       in a variable called self.cache. In practice, the
                       first time the data are read from the disk, they are
                       stored in the cpu RAM. If the data needs to be used
                       again (e.g. when loops>1) the data will be read
                       from the RAM directly. If False, data are read from
                       the disk every time.  Data are stored until a
                       certain percentage of the total ram available is
                       reached (see cache_ram_percent below)
        cache_ram_percent: If cache if True, data will be stored in the cpu
                     RAM until the total RAM occupation is less or equal
                     than the specified threshold. In practice, if a lot
                     of RAM is available several  data will be stored in
                     memory, otherwise, most of them will be read from the
                     disk directly.
        logger: it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.



     Input (call,__getitem__):
        idx: it is the index to read from the data list
                                   stored in data_dict['data_list'].

     Output (call,__getitem__):  - data: (type: list):
                                   it is a list containing the data. The list
                                   is formatted in the following way:
                                   [data_id,data,data_len]

     Example:  from speechbrain.data_io.data_io import create_dataloader
               from speechbrain.data_io.data_io import create_dataset

               config={'class_name':'core.loop',\
                       'csv_file':'samples/audio_samples/csv_example2.csv'
                       }

               # Initialization of the data_loader class
               data_loader=create_dataloader(config)

               # data_dict creation
               data_dict=data_loader.generate_data_dict()

               # supported formats
               formats=data_loader.get_supported_formats()

               # Initialization of the dataser class
               dataset=create_dataset(data_dict,{},formats,'wav',False,0)

               # Reading data
               print(dataset.__getitem__(0))
     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        data_dict,
        label_dict,
        supported_formats,
        data_entry,
        do_cache,
        cache_ram_percent,
        logger=None,
    ):

        # Setting the variables
        self.data_dict = data_dict
        self.label_dict = label_dict
        self.supported_formats = supported_formats
        self.logger = logger
        self.data_entry = data_entry

        self.do_cache = do_cache

        # Creating a shared dictionary for caching
        # (dictionary must be shared across the workers)
        if do_cache:
            manager = Manager()
            self.cache = manager.dict()
            self.cache["do_caching"] = True
            self.cache_ram_percent = cache_ram_percent

    def __len__(self):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.__len__ (author: Mirco Ravanelli)

         Description: This (mandatory) function returns the length of the
                      data list

         Args:
        self: Output:
        data_len: it is the number of data to read (i.e. len of the
                      data_list entry of the data_dict).

         Example:  from speechbrain.data_io.data_io import create_dataloader
                   from speechbrain.data_io.data_io import create_dataset

                   config={'class_name':'core.loop',\
                           'csv_file':'samples/audio_samples/csv_example2.csv'}

                   # Initialization of the data_loader class
                   data_loader=create_dataloader(config)

                   # data_dict creation
                   data_dict=data_loader.generate_data_dict()

                   # supported formats
                   formats=data_loader.get_supported_formats()

                   # Initialization of the dataser class
                   dataset=create_dataset(data_dict,{},formats,'wav',False,0)

                   # Getting data length
                   print(dataset.__len__())
         --------------------------------------------.------------------------
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

        # Reading data from data_dict
        data_line = self.data_dict[snt_id][self.data_entry]

        # Check if we need to convert labels to indexes
        if self.label_dict and self.data_entry in self.label_dict:
            lab2ind = self.label_dict[self.data_entry]["lab2index"]
        else:
            lab2ind = None

        # Managing caching
        if self.do_cache:

            if snt_id not in self.cache:

                # Reading data
                data = self.read_data(data_line, snt_id, lab2ind=lab2ind)

                # Store the in the variable cache if needed
                if self.cache["do_caching"]:

                    try:
                        self.cache[snt_id] = data
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
                data = self.cache[snt_id]
        else:
            # Read data from the disk
            data = self.read_data(data_line, snt_id, lab2ind=lab2ind)

        return data

    def read_data(self, data_line, snt_id, lab2ind=None):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.read_data (author: Mirco Ravanelli)

         Description: This function manages reading operation from disk.

         Args:
        self: - data_line (type: dict, mandatory):
                           it is one of entries extreacted from the data_dict.
                           It contains all the needed information to read the
                           data from the disk.
        snt_id: it the sentence identifier.

         Output:
        data_read: it is a list contaning the read data. The list if
                      formatted in the followig way: [data_id,data_data_len]

         Example:  from speechbrain.data_io.data_io import create_dataloader
                   from speechbrain.data_io.data_io import create_dataset

                   config={'class_name':'core.loop',\
                           'csv_file':'samples/audio_samples/csv_example2.csv'}

                   # Initialization of the data_loader class
                   data_loader=create_dataloader(config)

                   # data_dict creation
                   data_dict=data_loader.generate_data_dict()

                   # supported formats
                   formats=data_loader.get_supported_formats()

                   # Initialization of the dataser class
                   dataset=create_dataset(data_dict,{},formats,'wav',False,0)

                   # data line example
                   data_line={'data': 'samples/audio_samples/example5.wav', \
                              'format': 'wav',
                              'options': {'start': '10000', 'stop': '26000'}}
                   snt_id='example5'

                   # Reading data from disk
                   print(dataset.read_data(data_line,snt_id))
         --------------------------------------------.------------------------
         """

        # Reading the data_line dictionary
        data_format = data_line["format"]
        data_source = data_line["data"]
        data_options = data_line["options"]

        # Read the data from disk
        data = self.supported_formats[data_format]["reader"](
            data_source,
            data_options=data_options,
            lab2ind=lab2ind,
        )

        # Convert numpy array to float32
        if isinstance(data, np.ndarray):
            data_shape = np.asarray(data.shape[-1]).astype("float32")
        elif isinstance(data, torch.Tensor):
            data_shape = np.asarray(data.shape[-1]).astype("float32")

        else:
            data_shape = np.asarray(1).astype("float32")

        data_read = [snt_id, data, data_shape]

        return data_read


class save_ckpt(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_io.save_ckpt (author: Mirco Ravanelli)

     Description:
        This class can be use to save checkpoint during neural
        network training. It saves the current neural model,
        the current status of the optimizer, and stores the
        recovery information in the recovery dictionary.

     Args:
        - save_folder: str
           the folder where the recovery dict will be saved.

       - save_format: str
           it is the format used to save neural models.

       - save_last: int(0, inf)
           the maximum number of checkpoints to keep on disk.

       - write_result: bool
           when True, it save the performance on a res.res
           file in the output_folder.

     Example:
        >>> import torch
        >>> from speechbrain.data_io.data_io import load_pkl
        >>> performance_dict = {
        ...     'loss': torch.tensor([0.69]),
        ...     'error': torch.tensor([0.35]),
        ... }
        >>> save = save_ckpt(save_folder='exp/nnet')
        >>> save(0, performance_dict)
        >>> load_pkl('exp/nnet/recovery.pkl')
        {'loss': 0.69, 'error': 0.35}

     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        save_folder,
        save_format='pkl',
        save_last=1,
        write_result=True,
    ):
        super().__init__()

        def hook(self, input):
            perform_dict = input[1]

            if len(perform_dict) == 0:
                err_msg = (
                    "The second input to the function save_ckpt must be a "
                    "dict contaning at least one element (i.e, performance)"
                )
                logger.error(err_msg, exc_info=True)

            # Making sure that the list contains performance in tensor formats
            for perf in perform_dict:

                if isinstance(perform_dict[perf], list):
                    if len(perform_dict[perf]) == 1:
                        if isinstance(perform_dict[perf][0], torch.Tensor):
                            perform_dict[perf] = perform_dict[perf][0]

                if not isinstance(perform_dict[perf], torch.Tensor):
                    err_msg = (
                        "The second input to the function save_ckpt must be a "
                        "dict contaning torch.Tensor elements (Got %s)"
                    ) % (type(perform_dict[perf]))
                    logger.error(err_msg, exc_info=True)

        self.save_folder = save_folder
        self.save_format = save_format
        self.save_last = save_last
        self.write_result = write_result

        # Creating the save folder if it does not exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Initializing the best loss
        self.best_loss = None
        self.best_epoch = None

        # Check if the recovery file exists
        recovery_file = self.save_folder + "/recovery.pkl"

        if os.path.exists(recovery_file):

            # Loading the last best_loss/epoch saved
            self.recovery_dict = load_pkl(recovery_file)

            # if funct_name in self.recovery_dict:
            #    self.best_loss = self.recovery_dict[funct_name]["best_loss"]
            #    self.best_epoch = self.recovery_dict[funct_name]["best_epoch"]
        else:
            self.recovery_dict = {}

        if self.write_result:

            self.res_file = self.save_folder + "/res.res"

            if not os.path.exists(self.res_file):
                open(self.res_file, "a").close()

    def forward(self, epoch, performance_dict):
        """
        Args:
        epoch: the epoch label for record keeping
        performance_dict: a dictionary with the types of performance and
                    their corresponding values
        """
        # By default I use the lasr performance in input to decide the best
        # model.
        loss = performance_dict[list(performance_dict.keys())[-1]]

        # Converting loss to numpy array
        loss = loss.cpu().numpy()

        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = epoch

        if loss <= self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch

            # Saving the current best
            path = self.save_folder + "/current_best"
            self.save_neural_networks(path)

        # Write recovery info
        self.write_recovery_info(epoch, performance_dict)

        #  Saving current models
        path = self.save_folder + "/epoch_" + str(epoch) + "_loss_" + str(loss)
        self.save_neural_networks(path)

        # Save recovery info
        save_pkl(self.recovery_dict, self.save_folder + "/" + "recovery.pkl")

        # Delete old models
        if epoch - self.save_last >= 0:

            # Removing oldest models
            self.remove_model(epoch - self.save_last)

    def write_recovery_info(self, epoch, performance_dict):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.data_io.save_ckpt.write_recovery_info
         (auth: M. Ravanelli)

         Description: This function writes the performance dictionary on
                      disk.

         Args:
        epoch: it is the epoch id.
        performance_dict: it the recovery dictionary to store

         Output:    - None
                      the recovery dictionary is save in the save_folder
                      directory.

         Example:   import torch
                    from speechbrain.data_io.data_io import save_ckpt
                    from speechbrain.data_io.data_io import load_pkl

                    config={'class_name':'speechbrain.data_io.data_io.save_ckpt'}

                    # Initialization of the class
                    performance_dict={'loss':torch.tensor([0.69]),
                                      'error':torch.tensor([0.35])}

                    save=save_ckpt(config,global_config={'output_folder':\
                        'exp'},
                                     first_input=[0,performance_dict])


                    save.write_recovery_info(1,performance_dict)

                    # print recovery dictionary
                    print(load_pkl('exp/recovery.pkl'))
         --------------------------------------------.------------------------
         """

        # Path where previous performance dict is written
        recovery_file = self.save_folder + "/recovery.pkl"

        # check first entry of the performance dict
        loss = list(performance_dict.keys())[0]

        # Recovery previous recovery dict
        if os.path.exists(recovery_file):

            # Loading the last best_loss/epoch saved
            self.recovery_dict = load_pkl(recovery_file)

            # if self.funct_name not in self.recovery_dict:
            #    self.recovery_dict[self.funct_name] = {}

        # Creating a new recovery dictionary
        else:
            self.recovery_dict = {}

        # Saving recovery information in the recovery dictionary
        # self.recovery_dict[self.funct_name] = {
        #     "current_epoch": epoch,
        #     "current_loss": loss,
        #     "best_epoch": self.best_epoch,
        #     "best_loss": self.best_loss,
        # }

        # # Updating the recovery_dict with all the performance metrics
        # self.recovery_dict[self.funct_name].update(performance_dict)

        # Pring results (if needed)
        self.print_epoch(epoch, performance_dict)

    def save_neural_networks(self, path):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.data_io.save_ckpt.save_neural_networks
         (M. Ravanelli)

         Description: This function saves on disk all the parameters of
                      the neural networks found in the self.functions
                      dictionary.

         Args:
        path: it is the path where to store the neural models


         Output:    - None
                      the neural models are directly stored in the specified
                      path.


         Example:   import torch
                    from speechbrain.data_io.data_io import save_ckpt
                    from speechbrain.data_io.data_io import load_pkl
                    from speechbrain.nnet.architectures import linear

                    # Initializing a linear layer
                    inp_tensor = torch.rand([4,660,190])

                    config={'class_name':'speechbrain.nnet.architectures.linear',
                            'n_neurons':'1024'}

                    linear_transf=linear(config,first_input=[inp_tensor])

                    config={'class_name':'speechbrain.data_io.data_io.save_ckpt'}

                    # Initialization of the class
                    performance_dict={'loss':torch.tensor([0.69]),
                                      'error':torch.tensor([0.35])}

                    save=save_ckpt(config,global_config={'output_folder':\
                        'exp'},
                                     first_input=[0,performance_dict],
                                     functions={'linear':linear_transf})


                    save.save_neural_networks('exp/save_nn_exp')

                    # print recovery dictionary
                    print(torch.load('exp/save_nn_exp_linear.pkl'))
         --------------------------------------------.------------------------
         """
        """
        if self.functions is not None:

            # Loops over all the functions
            for funct in self.functions:

                # Check all the functions with the parameters attribute
                if hasattr(self.functions[funct], "parameters"):

                    # Save only model that actially have parameters
                    if len(list(self.functions[funct].parameters())) > 0:

                        path_full = path + "_" + funct
                        self.save_model(
                            self.functions[funct], path_full, funct
                        )

                # Saving optimizers
                if hasattr(self.functions[funct], "optim"):
                    path_full = path + "_" + funct
                    self.save_model(
                        self.functions[funct].optim, path_full, funct
                    )

                # Saving mean_norm statistics
                if hasattr(self.functions[funct], "mean_norm"):
                    path_full = path + "_" + funct
                    self.save_model(
                        self.functions[funct], path_full, funct, mean_stat=True
                    )
        """

    def save_model(self, model, path, funct, mean_stat=False):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.data_io.save_ckpt.save_model
         (author: Mirco Ravanelli)

         Description: This function saves on disk the pkl model

         Args:
        model: it is the model to save.
        path: it is the directory where storing the model.
        funct_name: it is the name of the neural model.
        mean_stat: this flag is True when the input model is just
                           a dictionary containing mean_statisics.


         Output:    - None
                      the neural models id directly save in the specified
                      path.


         Example:   import torch
                    from speechbrain.data_io.data_io import save_ckpt
                    from speechbrain.data_io.data_io import load_pkl
                    from speechbrain.nnet.architectures import linear

                    # Initializing a linear layer
                    inp_tensor = torch.rand([4,660,190])

                    config={'class_name':'speechbrain.nnet.architectures.linear',
                            'n_neurons':'1024'}

                    linear_transf=linear(config,first_input=[inp_tensor])

                    config={'class_name':'speechbrain.data_io.data_io.save_ckpt'}

                    # Initialization of the class
                    performance_dict={'loss':torch.tensor([0.69]),
                                      'error':torch.tensor([0.35])}

                    save=save_ckpt(config,global_config={'output_folder':\
                        'exp'},
                                     first_input=[0,performance_dict],
                                     functions={'linear':linear_transf})


                    save.save_model(linear_transf,'save_nn_example.pkl',\
                        'linear')

         --------------------------------------------.------------------------
         """

        # Managing pkl format
        if self.save_format == "pkl":

            path = path + ".pkl"

            # Check if model is mean_stat dictionary
            if mean_stat:
                torch.save(model.statistics_dict(), path)
            else:
                torch.save(model.state_dict(), path)

        # Update Recovery info:
        self.recovery_dict[self.funct_name][funct] = path

    def remove_model(self, epoch):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.data_io.save_ckpt.remove_model
         (author: Mirco Ravanelli)

         Description: This function support class removes old neural models
                      corresponding to previous epochs.

         Args:
        epoch: all the  neural models corresponding the specified
                         epoch will be removed from the save_folder.


         Output:    - None
                      the neural models are directly removed from the disk.


         Example:   import torch
                    from speechbrain.data_io.data_io import save_ckpt
                    from speechbrain.data_io.data_io import load_pkl
                    from speechbrain.nnet.architectures import linear

                    # Initializing a linear layer
                    inp_tensor = torch.rand([4,660,190])

                    config={'class_name':'speechbrain.nnet.architectures.linear',
                            'n_neurons':'1024'}

                    linear_transf=linear(config,first_input=[inp_tensor])

                    config={'class_name':'speechbrain.data_io.data_io.save_ckpt'}

                    # Initialization of the class
                    performance_dict={'loss':torch.tensor([0.69]),
                                      'error':torch.tensor([0.35])}

                    save=save_ckpt(config,global_config={'output_folder':\
                        'exp'},
                                     first_input=[0,performance_dict],
                                     functions={'linear':linear_transf})


                    save.save_neural_networks('exp/save_nn_exp')

                    # print recovery dictionary
                    print(torch.load('exp/save_nn_exp_linear.pkl'))

                    save.remove_model(0)
         --------------------------------------------.------------------------
         """

        # Loop over all files in the save_folder
        for file in os.listdir(self.save_folder):
            # Check if files containing the given epoch-id are present
            if "epoch_" + str(epoch) + "_" in file:
                os.remove(self.save_folder + "/" + file)

    def print_epoch(self, epoch, performance_dict):
        """
         ---------------------------------------------------------------------
         speechbrain.data_io.data_io.save_ckpt.print_epoch
         (author: Mirco Ravanelli)

         Description: This function prints the current performance in
                      the logger and in the res.res.file

         Args:
        epoch: it is the current epoch_id
        performance_dict: it is the dictionary containing the current
                           performance metrics.


         Output:    - None
                      the neural models are directly written in the log file
                      and in the res.res.file


         Example:   import torch
                    from speechbrain.data_io.data_io import save_ckpt
                    from speechbrain.data_io.data_io import load_pkl
                    from speechbrain.nnet.architectures import linear

                    # Initializing a linear layer
                    inp_tensor = torch.rand([4,660,190])

                    config={'class_name':'speechbrain.nnet.architectures.linear',
                            'n_neurons':'1024'}

                    linear_transf=linear(config,first_input=[inp_tensor])

                    config={'class_name':'speechbrain.data_io.data_io.save_ckpt'}

                    # Initialization of the class
                    performance_dict={'loss':torch.tensor([0.69]),
                                      'error':torch.tensor([0.35])}

                    save=save_ckpt(config,global_config={'output_folder':\
                        'exp'},
                                     first_input=[0,performance_dict],
                                     functions={'linear':linear_transf})


                    save.print_epoch(0,performance_dict)

                    # see exp/nnets/res.res

         --------------------------------------------.------------------------
         """

        if self.write_result:

            # Composing string to write
            string = "epoch %i:" % (epoch)

            # Printing all the metrics in the performance dict
            for perf in performance_dict:
                performance_dict[perf] = performance_dict[perf].cpu().numpy()
                string = " %s %s=%.4f" % (string, perf, performance_dict[perf])

            # Print learning rates
            # if self.functions is not None:

            #     for funct in self.functions:

            #         # Checking for optimization info
            #         if "optim" in self.functions[funct].__dict__:

            #             # Check the learning rate
            #             if "lr" in self.functions[funct].optim.param_groups[0]:

            #                 if (
            #                     "prev_lr"
            #                     in self.functions[funct].optim.param_groups[0]
            #                 ):
            #                     lr = self.functions[funct].optim.param_groups[
            #                         0
            #                     ]["prev_lr"]
            #                 else:
            #                     lr = self.functions[funct].optim.param_groups[
            #                         0
            #                     ]["lr"]

            #                 string = "%s lr_%s=%.8f" % (string, funct, lr)

            # Printing on logger
            logger.info(string)

            # Writing on the res.res file
            with open(self.res_file, "a") as file:
                file.write(string + "\n")


class print_predictions:
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_io.print_predictions (author: Mirco Ravanelli)

     Description: This class can be use to print in a csv file the
                   predictions performed by a neural network.
     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):

        # Logger Setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "ind2lab": ("string", "optional", "None"),
            "ctc_out": ("bool", "optional", "False"),
            "out_file": ("path", "optional", "None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, logger=self.logger
        )

        # Expected inputs when calling the class (no inputs in this case)
        self.expected_inputs = ["list", "torch.Tensor", "torch.Tensor"]

        # Checking the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Using index2lab dictionary from the label_dict
        self.lab_dict = global_config["label_dict"][self.ind2lab]["index2lab"]

        # if ctc_out is True, add blank symbol into the dictionary
        if self.ctc_out:
            self.lab_dict[len(self.lab_dict)] = "blank"

        # Set up file where to write predictions
        if self.out_file is None:
            self.out_file = global_config["output_folder"] + "/predictions.csv"

        self.file_pred = open(self.out_file, "w")

        msg = "\nPredictions:"
        logger_write(msg, logfile=self.logger, level="info")

        if len(self.lab_dict) > 0:
            msg = "id\t\tpredictions\tprobs"
            logger_write(msg, logfile=self.logger, level="info")
            self.file_pred.write("ID, duration,out,out_format,out_opts\n\n")
        else:
            msg = "id\t\\tpredictions"
            logger_write(msg, logfile=self.logger, level="info")
            self.file_pred.write("ID, duration,out,out_format,out_opts\n\n")

    def __call__(self, input_lst):

        # Reading input arguments
        ids, prob, prob_len = input_lst

        # Getting scores and predictions
        scores, predictions = torch.max(prob, dim=-2)

        # Opening the prediction file
        self.file_pred = open(self.out_file, "a")

        # Loop over all the input sentences
        for i, snt_id in enumerate(ids):

            # Getting current predictions
            current_pred = predictions[i]

            # Getting corrent scores (switch to linear prob)
            current_score = torch.exp(scores[i])

            # Score averaging
            current_score = current_score.mean()

            if len(predictions.shape) > 1:
                actual_size = int(
                    torch.round(prob_len[i] * predictions.shape[-1])
                )
                current_pred = current_pred[0:actual_size]

            if len(self.lab_dict) > 0:
                string_pred = []

                # Getting the label corresponding to the output index
                for pred_index in range(current_pred.shape[0]):
                    index_lab = int(current_pred[pred_index])

                    if index_lab in self.lab_dict:
                        string_pred.append(self.lab_dict[index_lab])

                    else:

                        err_msg = (
                            "The index %i is not found in the self.lab_dict "
                            "for the lab %s. This might happen when some labs "
                            "are present in the test set, but not in the "
                            "training set. Make sure your train set contains "
                            "all the needed labels. It might happens when "
                            "the fields lab2index= and lab2index_dict= contain"
                            "wrong labels or path to different dictionary. "
                            % (index_lab, self.ind2lab)
                        )

                        logger.error(err_msg, exc_info=True)

            if len(self.lab_dict) > 0:
                if len(string_pred) == 1:
                    string_pred = string_pred[0]

                if self.ctc_out:

                    # Filtering the ctc output predictions
                    string_pred = filter_ctc_output(
                        string_pred, blank_id="blank"
                    )

                # Converting list to string
                len_str = len(string_pred)
                if isinstance(string_pred, list):
                    string_pred = " ".join(str(x) for x in string_pred)

                    # Filtering sil sil patterns
                    string_pred = string_pred.replace("sil sil", "sil")

                # Writing the output
                msg = "%s\t%s\t%.3f" % (snt_id, string_pred, current_score)
                logger_write(msg, logfile=self.logger, level="info")

                self.file_pred.write(
                    "%s,%i,%s,string,\n" % (snt_id, len_str, string_pred)
                )

            else:
                # Writing output
                msg = "%s\t%i" % (snt_id, current_pred)
                logger_write(msg, logfile=self.logger, level="info")
                self.file_pred.write(
                    "%s,1.0,%i,string,\n" % (snt_id, current_pred)
                )

        # Closing the prediction file
        self.file_pred.close()


def filter_ctc_output(string_pred, blank_id=-1, logger=None):
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_io.filter_ctc_output (author: Mirco Ravanelli)

     Description: This filters the output of a ctc-based system by removing
                  the blank symbol and the output repetitions. The function
                  also accept in input tensors of integer indexes.


     Input (call): - string_pred(type, list, mandatory):
                       it is a list contaning the output strings predicted
                       by the CTC system.


     Output (call): - string_out(type, list):
                       It is a list returning the output predicted by CTC
                       without the blank symbol and the repetitions

     Example:   from speechbrain.data_io.data_io import filter_ctc_output

                string_pred = ['a','a','blank','b','b','blank','c']

                string_out = filter_ctc_output(string_pred, blank_id='blank')

                print(string_out)
     --------------------------------------------.----------------------------
     """

    if isinstance(string_pred, list):
        # Filterning the repetitions
        string_out = [
            v
            for i, v in enumerate(string_pred)
            if i == 0 or v != string_pred[i - 1]
        ]

        # Removing duplicates
        string_out = [i[0] for i in groupby(string_out)]

        # Filterning the blank symbol
        string_out = list(filter(lambda elem: elem != blank_id, string_out))

    return string_out


def recovery(self):
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_io.recovery(author: Mirco Ravanelli)

     Description: when the recovery dictionary exists, this function
                  loads the saved parameters into the corresponding
                  neural networks. The function is used to re-start
                  the neural network training from the last epoch
                  correctly saved.
     --------------------------------------------.----------------------------
     """

    if self.recovery:

        recovery_file = self.output_folder + "/recovery.pkl"

        if os.path.exists(recovery_file):

            # Reading the recovery dictionary to recover file
            recovery_dict = load_pkl(recovery_file)

            # finding all the entries in the recovery_dict
            for key, value in recursive_items(recovery_dict):

                if key == self.funct_name:

                    if self.recovery_type == "last":
                        model_file = value

                    if self.recovery_type == "best":

                        # Getting file of the best model
                        model_folder = "/".join(value.split("/")[:-1])
                        model_name = value.split("/")[-1]
                        model_name = "_".join(model_name.split("_")[4:])
                        model_name = "current_best_" + model_name
                        model_file = model_folder + "/" + model_name

                    # Loading the model
                    if hasattr(self, "optim"):
                        self.optim.load_state_dict(torch.load(model_file))

                    elif hasattr(self, "mean_norm"):
                        self.load_statistics_dict(torch.load(model_file))

                    else:
                        self.load_state_dict(torch.load(model_file))

                    text = "Loaded model %s" % (model_file)
                    logger_write(text, logfile=self.logger, level="debug")

                    break


def initialize_with(self):
    """
     -------------------------------------------------------------------------
     speechbrain.data_io.data_io.initialize_with(author: Mirco Ravanelli)

     Description: This function initialize the parameters of the neural
                  network with the pkl file reported in the field
                  initialized_with
     --------------------------------------------.----------------------------
    """

    if self.initialize_with is not None and self.initialize_with != "random":

        # Check if the specified file exists:
        if not os.path.isfile(self.initialize_with):
            err_msg = (
                'the file specified in the field "initialize_with"'
                "does not exist"
            )

            logger.error(err_msg, exc_info=True)

        # Loading the initialization parameters
        self.load_state_dict(torch.load(self.initialize_with))


def read_wav_soundfile(file, data_options={}, lab2ind=None):
    """
    -------------------------------------------------------------------------
    Description:
        This function reads audio files with soundfile.

    Args:
        file: it is the file to read.
        data_options: a dictionary containing options for the reader.
        lab2ind: a dictionary for converting labels to indices

    Returns:
        An array with the read signal

    Example:
        >>> read_wav_soundfile('samples/audio_samples/example1.wav')

    Author:
        Mirco Ravanelli 2020
     -------------------------------------------------------------------------
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

    except Exception:
        err_msg = "cannot read the wav file %s" % (file)
        logger.error(err_msg, exc_info=True)

    # Set time_steps always last as last dimension
    if len(signal.shape) > 1:
        signal = signal.transpose()

    return signal


def read_pkl(file, data_options={}, logger=None, lab2ind=None):
    """
     -------------------------------------------------------------------------
     data_io.read_pkl (author: Mirco Ravanelli)

     Description: This function reads tensors store in pkl format.

     Input (call):
        file: it is the file to read.

                       - data_options(type: dict, mandatory):
                           it is a dictionary containing options for the
                           reader.
        logger: it the logger used to write debug and error messages.


     Output (call):  tensor (type: numpy.array):
                       it is the array containing the read signal.


     Example:  from speechbrain.data_io.data_io import read_pkl

               print(read_pkl('pkl_file.pkl'))

     -------------------------------------------------------------------------
     """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            pkl_element = pickle.load(f)
    except Exception:
        err_msg = "cannot read the pkl file %s" % (file)
        logger.error(err_msg, exc_info=True)

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
            err_msg = "The pkl file %s can only contain list of integers, "
            "floats, or strings. Got %s" % (file, type(pkl_element[0]))
            logger.error(err_msg, exc_info=True)
    else:
        tensor = pkl_element

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int32")

    return tensor


def read_string(string, data_options={}, logger=None, lab2ind=None):
    """
     -------------------------------------------------------------------------
     data_io.read_string (author: Mirco Ravanelli)

     Description: This function reads data in string format.

     Input (call):
        string: it is the string to read

                       - data_options(type: dict, mandatory):
                           it is a dictionary containing options for the
                           reader.
        logger: it the logger used to write debug and error messages.


     Output (call):  tensor (type: numpy.array):
                       it is the array containing the read signal.


     Example:  from speechbrain.data_io.data_io import read_string

               print(read_string('hello_world'))

     -------------------------------------------------------------------------
     """

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
     -------------------------------------------------------------------------
     data_io.read_kaldi_lab (author: Mirco Ravanelli)

     Description: This function reads label in kaldi format

     Input (call):
        kaldi_ali: it is the directory where kaldi alignents are
                           stored.

                       - kaldi_lab_opts(type: str, mandatory):
                           it is a string that contains the options for
                           reading the kaldi alignments.
        logfile: it the logger used to write debug and error msgs.


     Output (call):  lab (type: dict):
                       it is a dictionary contaning the labels


     Example:  from speechbrain.data_io.data_io import read_kaldi_lab

               lab_folder='/home/kaldi/egs/TIMIT/s5/exp\
               /dnn4_pretrain-dbn_dnn_ali'
               print(read_kaldi_lab(lab_folder,'ali-to-pdf'))

     -------------------------------------------------------------------------
     """

    # Reading the Kaldi labels
    lab = {
        k: v
        for k, v in read_vec_int_ark(
            "gunzip -c "
            + kaldi_ali
            + "/ali*.gz | "
            + kaldi_lab_opts
            + " "
            + kaldi_ali
            + "/final.mdl ark:- ark:-|",
            logfile=logfile,
        )
    }
    return lab


def write_wav_soundfile(data, filename, sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.write_wav_soundfile (author: Mirco Ravanelli)

     Description: This function can be used to write audio with soundfile.

     Input (call):
        data: it is the tensor to store as and audio file
        filename: it is the file where writign the data.
        sampling_rate: it sampling rate of the audio file.
        logger: it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from speechbrain.data_io.data_io import write_wav_soundfile

               signal=0.1*torch.rand([16000])
               write_wav_soundfile(signal,'exp/wav_example.wav',
               sampling_rate=16000)

     -------------------------------------------------------------------------
     """

    # Switch from (channel,time) to (time,channel) as Expected
    if len(data.shape) > 2:

        err_msg = (
            "expected signal in the format (channel,time). Got %s "
            "dimensions instead of two for file %s"
            % (len(data.shape), filename)
        )

        logger.error(err_msg, exc_info=True)

    if isinstance(data, torch.Tensor):

        if len(data.shape) == 2:
            data = data.transpose(0, 1)

        # Switching to cpu and converting to numpy
        data = data.cpu().numpy()

    # Writing the file
    try:
        sf.write(filename, data, sampling_rate)
    except Exception:
        err_msg = "cannot write the wav file %s" % (filename)
        logger.error(err_msg, exc_info=True)


def write_txt_file(data, filename, sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.write_txt_file (author: Mirco Ravanelli)

     Description: This function write data in text format

     Input (call):
        data: mandatory):
                           it is the data to write in the text file
        filename: it is the file where writing the data.
        sampling_rate: it sampling rate of the audio file.
        logger: it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from speechbrain.data_io.data_io import write_txt_file

               signal=torch.tensor([1,2,3,4])
               write_txt_file(signal,'exp/example.txt')

     -------------------------------------------------------------------------
     """

    # Check if the path of filename exists
    if not os.path.exists(os.path.dirname(filename)):

        try:
            os.makedirs(os.path.dirname(filename))
        except Exception:
            err_msg = "cannot create the file %s." % (filename)
            logger.error(err_msg, exc_info=True)

    # Opening the file
    try:
        file_id = open(filename, "w")

    except Exception:
        err_msg = "cannot create the file %s." % (filename)
        logger.error(err_msg, exc_info=True)

    # Managing torch.Tensor
    if isinstance(data, torch.Tensor):
        data = data.tolist()

    # Managing np.ndarray
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # Managing list
    if isinstance(data, list):
        for line in data:
            print(line, file=file_id)

    # Managing str
    if isinstance(data, str):
        print(data, file=file_id)

    # Closing the file
    file_id.close()


def write_stdout(data, filename, sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.write_txt_file (author: Mirco Ravanelli)

     Description: This function write data to standar output

     Input (call):
        data: mandatory):
                           it is the data to write in the text file
        filename: it is the file where writing the data.
        sampling_rate: it sampling rate of the audio file.
        logger: it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from speechbrain.data_io.data_io import write_stdout

               signal=torch.tensor([1,2,3,4])
               write_stdout(signal,'exp/example.txt')

     -------------------------------------------------------------------------
     """

    # Managing Torch.Tensor
    if isinstance(data, torch.Tensor):
        data = data.tolist()

    # Managing np.ndarray
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # Writing to stdout
    print(data, file=sys.stdout)


def save_img(data, filename, sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io. save_img (author: Mirco Ravanelli)

     Description: This function save a tensor as an image.

     Input (call):
        data: it is the tensor to write in the text file
        filename: it is the file where writing the data.
        sampling_rate: it sampling rate of the audio file.
        logger: it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from data_io import save_img

               signal=torch.rand([100,200])
               save_img(signal,'exp/example.png')

     -------------------------------------------------------------------------
     """
    # If needed import matplotlib
    try:

        import matplotlib.pyplot as plt

    except Exception:

        err_msg = "cannot import matplotlib. Make sure it is installed." % (
            filename,
            str(data.shape),
        )

        logger.error(err_msg, exc_info=True)
        raise

    # Checking tensor dimensionality
    if len(data.shape) < 2 or len(data.shape) > 3:

        err_msg = (
            "cannot save image  %s. Save in png format supports 2-D or "
            "3-D tensors only (e.g. [x,y] or [channel,x,y]). "
            "Got %s" % (filename, str(data.shape))
        )

        logger.error(err_msg, exc_info=True)

    if len(data.shape) == 2:
        N_ch = 1
    else:
        N_ch = data.shape[0]

    # Flipping axis
    data = data.flip([-2])

    for i in range(N_ch):

        if N_ch > 1:
            filename = filename.replace(".png", "_ch_" + str(i) + ".png")

        # Saving the image
        try:
            if N_ch > 1:
                plt.imsave(filename, data[i])
            else:
                plt.imsave(filename, data)
        except Exception:
            err_msg = "cannot save image  %s." % (filename)
            logger.error(err_msg, exc_info=True)


class save(torch.nn.Module):
    """
    -------------------------------------------------------------------------
    Description:
       This class can be used to save tensors on disk.

    Args:
        save_folder: it is the folder where the tensors are stored.
        save_format: it is the format to use to save the tensor.
            See get_supported_formats() for an overview of
            the supported data formats.
        save_csv: if True it saves the list of data written in a
            csv file.
        data_name: it is the name to give to saved data
        parallel_write: if True it saves the data using parallel processes.
        transpose: if True it transposes the data matrix
        decibel: if True it saves the log of the data.

     Example:
        >>> import torch
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> signal = 0.1 * torch.rand([1, 16000])
        >>> save_signal(signal, ['example_random'], torch.ones(1))

    Author:
        Mirco Ravanelli 2020
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        save_folder=None,
        save_format='pkl',
        save_csv=False,
        data_name='data',
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
            self.save_csv_path = self.save_folder + "/csv.csv"
            open(self.save_csv_path, "w").close()
            self.first_line_csv = True

    def forward(self, data, data_id, data_len):
        """
        Input :
        data: batch of audio signals to save
        data_id: list of ids in the batch
        data_len: length of each audio signal
        """
        # Convertion to log (if specified)
        if self.decibel:
            data = 10 * data.log10()

        # Writing data on disk (in parallel)
        self.write_batch(data, data_id, data_len)

    def write_batch(self, data, data_id, data_len):
        """
         ---------------------------------------------------------------------
         core.data_procesing.save.write_batch

         Description: This function saves a batch of data.

         Args:
        self: - data (type: torch.tensor, mandatory)
        data_id: - data_len (type: torch.tensor, mandatory)

         Output:      None

         Example:  import torch
                   from speechbrain.data_io.data_io import save

                   # save config dictionary definition
                   config={'class_name':'data_processing.save',
                           'save_folder': 'exp/write_example',
                           'save_format': 'wav' }

                   # class initialization
                   save_signal=save(config)

                   # random signal
                   signal=0.1*torch.rand([1,16000])

                   # saving
                   save_signal.write_batch(signal,['example_random'],
                   torch.ones(1))

                  # signal save in exp/write_example
         ---------------------------------------------------------------------
         """

        # Write in parallel all the examples in the batch on disk:
        jobs = []

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
            actual_size = int(torch.round(data_len[j] * data[j].shape[-1]))
            data_save = data[j].narrow(-1, 0, actual_size)

            # Transposing the data if needed
            if self.transpose:
                data_save = data_save.transpose(-1, -2)

            # Selection of the needed data writer
            writer = self.supported_formats[self.save_format]["writer"]

            # Output file
            data_file = (
                self.save_folder + "/" + data_id[j] + "." + self.save_format
            )

            # Writing all the batches in parallel (if paralle_write=True)
            if self.parallel_write:
                p = mp.Process(
                    target=writer,
                    args=(data_save, data_file),
                    kwargs={
                        "sampling_rate": self.sampling_rate,
                        "logger": self.logger,
                    },
                )
                p.start()
                jobs.append(p)
            else:
                # Writing data on disk with the selected writer
                writer(
                    data_save,
                    data_file,
                    sampling_rate=self.sampling_rate,
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
         ---------------------------------------------------------------------
         core.data_procesing.save.get_supported_formats

         Description: This function returns a dictionay containing the
                      supported writing format and the related writers
                      implemented in data_io.py.

         Args:
        self: Output:      -supported_formats (type:dict)

         Example:  import torch
                   from speechbrain.data_io.data_io import save

                   # save config dictionary definition
                   config={'class_name':'data_processing.save',
                           'save_folder': 'exp/write_example',
                           'save_format': 'wav' }

                   # class initialization
                   save_signal=save(config)

                   supported_formats=save_signal.get_supported_formats()
                   print(supported_formats)
         ---------------------------------------------------------------------
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

        supported_formats["ark"] = {}
        supported_formats["ark"]["writer"] = write_ark
        supported_formats["ark"]["description"] = "Kaldi binary format"

        supported_formats["png"] = {}
        supported_formats["png"]["writer"] = save_img
        supported_formats["png"]["description"] = "image in png format"

        supported_formats["stdout"] = {}
        supported_formats["stdout"]["writer"] = write_stdout
        supported_formats["stdout"]["description"] = "write on stdout"

        return supported_formats


def write_ark(data, filename, key="", sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.write_ark (author: github.com/vesis84/kaldi-io-for-python)

     Description: write_mat(data,filename key='')
                  Write a binary kaldi matrix to filename or stream.
                  Supports 32bit and 64bit floats.

     Input (call):
        filename: it is filename of opened file descriptor for writing.
        data: it is the matrix to be stored
        key: it is used for writing ark-file, the utterance-id gets
                       written before the matrix.
        sampling_rate: it sampling rate of the audio file.
        logger: it the logger used to write debug and error messages.

     Output (call):  None

     Example:   import torch
                from speechbrain.data_io.data_io import write_ark

                matrix = torch.rand([5,10])

                write_ark(matrix,'matrix.ark',key='snt')
     -------------------------------------------------------------------------
     """

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    fd = open_or_fd(filename, None, mode="wb")

    # Detecting sentence id to write
    key = os.path.basename(filename).split(".")[0]

    if sys.version_info[0] == 3:
        assert fd.mode == "wb"
    try:
        if key != "":
            # ark-files have keys (utterance-id),
            fd.write((key + " ").encode("latin1"))
        fd.write("\0B".encode())  # we write binary!

        # Data-type,
        if data.dtype == "float32":
            fd.write("FM ".encode())
        elif data.dtype == "float64":
            fd.write("DM ".encode())
        else:
            raise UnsupportedDataType(
                "'%s', please use 'float32' \
                                      or 'float64'"
                % data.dtype
            )
        # Dims,
        fd.write("\04".encode())
        fd.write(struct.pack(np.dtype("uint32").char, data.shape[0]))  # rows
        fd.write("\04".encode())
        fd.write(struct.pack(np.dtype("uint32").char, data.shape[1]))  # cols
        # Data,
        fd.write(data.tobytes())
    finally:
        if fd is not filename:
            fd.close()


def get_md5(file):
    """
     -------------------------------------------------------------------------
     data_io.get_md5 (author: Mirco Ravanelli)

     Description: This function return the md5 checksum of an input file

     Input (call):
        file: it is the file from which we want to computed the
                           md5


     Output (call):  md5 (type: md5):
                       it is the checksum for the given file


     Example:  from speechbrain.data_io.data_io import  get_md5
               print(get_md5('samples/audio_samples/example1.wav'))

     -------------------------------------------------------------------------
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
     -------------------------------------------------------------------------
     data_io.save_md5 (author: Mirco Ravanelli)

     Description: This function saves the md5 of a list of input files into
                   a dictionary. The dictionary is then save in pkl format.

     Input (call):
        files: it is a list of input files from which we will
                           compute the md5.
        outfile: it is the path where storing the output pkl file.

     Output (call):  None


     Example:  from speechbrain.data_io.data_io import save_md5

               files=['samples/audio_samples/example1.wav']
               out_file='exp/md5.pkl'
               save_md5(files,out_file)

     -------------------------------------------------------------------------
     """

    # Initialization of the dictionary
    md5_dict = {}

    # Computing md5 for all the files in the list
    for file in files:
        md5_dict[file] = get_md5(file)

    # Saving dictionary in pkl format
    save_pkl(md5_dict, out_file)


def save_pkl(obj, file, sampling_rate=None, logger=None):
    """
    -------------------------------------------------------------------------
    Description: This function saves an object in pkl format.

    Args:
        obj: it object to save in pkl format
        file: it is name of the output file.
        sampling_rate: it sampling rate of the audio file.
        logger: it the logger used to write debug and error messages.

    Example:
        >>> out_file='exp/example.pkl'
        >>> save_pkl([1, 2, 3, 4, 5], out_file)
        >>> load_pkl(out_file)
        [1, 2, 3, 4, 5]

    Author:
        Mirco Ravanelli 2020
    -------------------------------------------------------------------------
    """
    try:
        with open(file, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        err_msg = "Cannot save file %s" % (file)
        logger.error(err_msg, exc_info=True)


def load_pkl(file, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.load_pkl (author: Mirco Ravanelli)

     Description:
        This function load a pkl file

     Args:
        file: it is name of the input pkl file.
        logger: it the logger used to write debug and error messages.

     Returns:
        the loaded object


     Example:
        >>> pkl_file='exp/example.pkl'
        >>> save_pkl([1, 2, 3, 4, 5], pkl_file)
        >>> load_pkl(pkl_file)
        [1, 2, 3, 4, 5]

     -------------------------------------------------------------------------
     """

    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except Exception:
        err_msg = "Cannot read file %s" % (file)
        logger.error(err_msg, exc_info=True)


def read_vec_int_ark(file_or_fd):
    """
     -------------------------------------------------------------------------
     data_io.read_vec_int_ark (author: github.com/vesis84/kaldi-io-for-python)
     -------------------------------------------------------------------------
     """

    fd = open_or_fd(file_or_fd, logfile)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_int(fd, logfile)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_key(fd):
    """
     -------------------------------------------------------------------------
     data_io.read_key (author: github.com/vesis84/kaldi-io-for-python)

     Description: Read the utterance-key from the opened ark/stream descriptor
                  'fd'.
     -------------------------------------------------------------------------
     """

    key = ""
    while 1:
        char = fd.read(1).decode("latin1")
        if char == "":
            break
        if char == " ":
            break
        key += char
    key = key.strip()
    if key == "":
        return None  # end of file,
    assert re.match(r"^\S+$", key) is not None  # check format (no white space)
    return key


def read_vec_int(file_or_fd):
    """
     -------------------------------------------------------------------------
     data_io.read_key (author: github.com/vesis84/kaldi-io-for-python)

     Description: Read kaldi integer vector, ascii or binary input,
     -------------------------------------------------------------------------
     """

    fd = open_or_fd(file_or_fd, logfile=logfile)
    binary = fd.read(2).decode()
    if binary == "\0B":  # binary flag
        assert fd.read(1).decode() == "\4"
        # int-size
        vec_size = np.frombuffer(fd.read(4), dtype="int32", count=1)[0]
        if vec_size == 0:
            return np.array([], dtype="int32")
        # Elements from int32 vector are stored in tuples:
        # (sizeof(int32), value),
        vec = np.frombuffer(
            fd.read(vec_size * 5),
            dtype=[("size", "int8"), ("value", "int32")],
            count=vec_size,
        )
        assert vec[0]["size"] == 4  # int32 size,
        ans = vec[:]["value"]  # values are in 2nd column,
    else:  # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove("[")
            arr.remove("]")  # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    if fd is not file_or_fd:
        fd.close()  # cleanup
    return ans


def open_or_fd(file, mode="rb"):
    """
     -------------------------------------------------------------------------
     data_io.open_or_fd (author: github.com/vesis84/kaldi-io-for-python)

     Description: Open file, gzipped file, pipe, or forward the
                  file-descriptor.
                  Eventually seeks in the 'file' argument contains ':offset'
                  suffix.
     -------------------------------------------------------------------------
     """

    offset = None

    try:
        # Strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search(
            "^(ark|csv)(,csv|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:", file
        ):
            (prefix, file) = file.split(":", 1)
        # Separate offset from filename (optional),
        if re.search(":[0-9]+$", file):
            (file, offset) = file.rsplit(":", 1)
        # Input pipe?
        if file[-1] == "|":
            fd = popen(file[:-1], logfile=logfile, mode="rb")  # custom,
        # Output pipe?
        elif file[0] == "|":
            fd = popen(file[1:], logfile=logfile, mode="wb")  # custom,
        # Is it gzipped?
        elif file.split(".")[-1] == "gz":
            fd = gzip.open(file, mode)
        # A normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset is not None:
        fd.seek(int(offset))

    return fd


def popen(cmd, mode="rb"):
    """
     -------------------------------------------------------------------------
     data_io.popen (author: github.com/vesis84/kaldi-io-for-python)
     -------------------------------------------------------------------------
     """

    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

    # cleanup function for subprocesses,
    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise SubprocessFailed("cmd %s returned %d !" % (cmd, ret))
        return

    if logfile is not None:
        err = open(logfile, "a")
    else:
        err = None

    # text-mode,
    if mode == "r":
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=err
        )
        threading.Thread(
            target=cleanup, args=(proc, cmd)
        ).start()  # clean-up thread,
        return io.TextIOWrapper(proc.stdout)
    elif mode == "w":
        proc = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, stderr=err
        )
        threading.Thread(
            target=cleanup, args=(proc, cmd)
        ).start()  # clean-up thread,
        return io.TextIOWrapper(proc.stdin)
    # binary,
    elif mode == "rb":
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=err
        )
        threading.Thread(
            target=cleanup, args=(proc, cmd)
        ).start()  # clean-up thread,
        return proc.stdout
    elif mode == "wb":
        proc = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, stderr=err
        )
        threading.Thread(
            target=cleanup, args=(proc, cmd)
        ).start()  # clean-up thread,
        return proc.stdin
    # sanity,
    else:
        raise ValueError("invalid mode %s" % mode)


class SubprocessFailed(Exception):
    """
    -------------------------------------------------------------------------
    data_io.SubprocessFailed (author: github.com/vesis84/kaldi-io-for-python)
    -------------------------------------------------------------------------
    """

    pass


class UnsupportedDataType(Exception):
    """
    -------------------------------------------------------------------------
    data_io.UnsupportedDataType (github.com/vesis84/kaldi-io-for-python)
    -------------------------------------------------------------------------
    """

    pass
