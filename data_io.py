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
import sys
import gzip
import torch
import struct
import psutil
import random
import pickle
import hashlib
import threading
import subprocess
import numpy as np
import soundfile as sf
from torch.multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
from utils import logger_write, check_opts, remove_comments


class create_dataloader:
    """
     -------------------------------------------------------------------------
     data_io.create_dataloader (author: Mirco Ravanelli)

     Description: This class creates the data_loaders for the given scp file.

     Input (init):  - exec_config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - scp (type: file, mandatory):
                               it is the scp file that itemized the data.

                           - batch_size: (type: int(1,inf),optional,
                               default: 1):
                               the data itemized in the scp file are
                               automatically organized in batches. In the case
                               of variable size tensors, zero padding is
                               performed. When batch_size=1, the data are
                               simply processed one by one without the
                               creation of batches.

                           - scp_read (type: str_list,optional,default:None):
                               this option can be used to read only some
                               data_entries of the scp file. When not
                                specified, it automatically reads all the data
                                entries.

                           - sentence_sorting: ('ascending,descending,random,
                             original', optional, 'original'):
                               This parameter specifies how to sort the data
                               before the batch creation. Ascending and
                               descending values sort the data using the
                               "duration" field in the scp files. Random sort
                               the data randomly, while original (the default
                               option) keeps the original sequence of data
                               defined in the scp file. Note that this option
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

                           - select_n_sentences (type: int(1,inf),
                             optional,None):
                               this option can be used to read-only n
                               sentences from the scp file. This option can be
                               useful to debug the code, when instead of
                               running an experiment of a full set of data I
                               might just want to run it with a little about
                               of data.

                           - num_workers (int(0,inf),optional,Default:0):
                               data are read using the pytorch data_loader.
                               This option set the number of workers used to
                               read the data from disk and form the related
                               batch of data. Please, see the pytorch
                               documentation on the data loader for more
                               details.

                           - cache(bool,optional,Default:False):
                               When set to true, this option stores the input
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

                           - cache_ram_percent (int(0,100),optional,
                             default:75):
                               If cache if True, data will be stored in the
                               cpu RAM until the total RAM occupation is less
                               or equal than the specified threshold
                               (by default 75%). In practice, if a lot of RAM
                               is available several data will be stored in
                               memory, otherwise, most of them will be read
                               from the disk directly.

                           - drop_last (bool,optional,Default: False):
                               this is an option directly passed to the
                               pytorch dataloader (see the related
                               documentation for more details). When True,
                               it skips the last batch of data if contains
                               fewer samples than the other ones.

                   - funct_name (type, str, optional, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, optional, default: None)
                      this variable allows users to analyze the first input
                       given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       by default the input arguments are passed with a list.
                       In this case, the list gathers input variables that can
                       be used in the computations.


     Output (call):  - dataloader (type: dataloader):
                       It is a list returning all the dataloaders created.

     Example:   from data_io import create_dataloader

                config={'class_name':'core.loop',\
                         'scp':'samples/audio_samples/scp_example2.scp'}

                # Initialization of the class
                data_loader=create_dataloader(config)

                print(data_loader([]))
     --------------------------------------------.----------------------------
     """

    def __init__(
        self, config, funct_name=None, global_config=None, logger=None
    ):

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "scp": ("file", "mandatory"),
            "batch_size": ("int(1,inf)", "optional", "1"),
            "scp_read": ("str_list", "optional", "None"),
            "sentence_sorting": (
                "one_of(ascending,descending,random,original)",
                "optional",
                "original",
            ),
            "num_workers": ("int(0,inf)", "optional", "0"),
            "cache": ("bool", "optional", "False"),
            "cache_ram_percent": ("int(0,100)", "optional", "75"),
            "select_n_sentences": ("int(1,inf)", "optional", "None"),
            "drop_last": ("bool", "optional", "False"),
            "padding_value": ("int(-inf,inf)", "optional", "0"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(self, self.expected_options, config)

        # Other variables
        self.global_config = global_config
        self.logger = logger
        self.supported_formats = self.get_supported_formats()
        self.padding_value = 0

        # Shuffle the data every time if random is selected
        if self.sentence_sorting == "random":
            self.shuffle = True
        else:
            self.shuffle = False

        # create data dictionary
        data_dict = self.generate_data_dict()

        if self.scp_read is None:
            self.scp_read = data_dict["data_entries"]

        self.dataloader = []

        # Creating a dataloader for each data entry in the scp file
        for data_entry in self.scp_read:

            dataset = create_dataset(
                data_dict,
                self.supported_formats,
                data_entry,
                self.cache,
                self.cache_ram_percent,
                logger=self.logger,
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

    def __call__(self, inp):
        return [self.dataloader]

    def batch_creation(self, data_list):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.batch_creation (author: Mirco Ravanelli)

         Description: This function create the batch of data. When necessary
                      it performs zero padding on the input tensors. The
                      function is executed in collate_fn of the pytorch
                      DataLoader.

         Input:       - self (type:  create_dataloader class, mandatory)

                      - data_list (type: list, mandatory):
                        it is the list of data returned by the data reader
                        [data_id,data,data_len]

         Output:      - batch (type: list):
                        it is a list containing the final batches:
                        [data_id,data,data_len] where zero-padding is
                        performed where needed.

         Example:   from data_io import create_dataloader

                    config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

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

         Input:       - self (type:  create_dataloader class, mandatory)

                      - sequences (type: list, mandatory):
                        it is the list of tensor to pad.

         Output:      - batch_data (type: torch.Tensor):
                        it is a tensor gathering all the padded tensors.

         Example:   import torch
                    from data_io import create_dataloader

                    config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

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

         Input:       - self (type:  create_dataloader class, mandatory)

                      - data_list (type: list, mandatory):
                        it is a list of numpy arrays.

         Output:    None

         Example:  import numpy as np
                   from data_io import create_dataloader

                   config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

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

    def generate_data_dict(self,):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.generate_data_dict(author: Mirco Ravanelli)

         Description: This function creates a dictionary from the scp file

         Input:       - self (type:  create_dataloader class, mandatory)


         Output:    - data_dict (type: dict):
                       it is a dictionary with the data itemized in the scp
                       file.

         Example:  from data_io import create_dataloader

                   config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

                   # Initialization of the class
                   data_loader=create_dataloader(config)

                   print(data_loader.generate_data_dict())
         --------------------------------------------.------------------------
         """

        # Initial prints
        msg = "\tCreating dataloader for %s" % (self.scp)
        logger_write(msg, logfile=self.logger, level="debug")

        # Initialization of the data_dict
        data_dict = {}

        # Definition or patters to search into the scp file.
        value_regex = re.compile(r"([\w]*)=([\w\$\(\)\/'\"\,\-\_\.\:\#]*)")
        del_spaces = re.compile(r"=([\s]*)")
        extr_brakets = re.compile(r"\((.*?)\)")

        data_count = 0
        total_duration = 0

        data_entries = []

        # Opening the scp file.
        with open(self.scp) as scp:

            # Processing all the lines of the scp file.
            for data_line in scp:

                id_field = False
                duration_field = False
                first_item = True

                # Removing spaces
                data_line = data_line.strip()

                # Removing comments
                data_line = remove_comments(data_line)

                # Skipping the line if is empty.
                if len(data_line) == 0:
                    continue

                # Replacing multiple spaces.
                data_line = (
                    re.sub(" +", " ", data_line)
                    .replace(" ,", ",")
                    .replace(", ", ",")
                    .replace("= ", "=")
                    .replace(" =", "=")
                )

                # Extracting key=value patterns in the scp file
                data_line = del_spaces.sub("=", data_line)
                values = value_regex.findall(data_line)

                # Check if at least one key=value pattern exists
                if len(values) == 0:

                    err_msg = (
                        'the lines of the scp file %s should '
                        'contain key=value items (got %s)'
                        % (self.scp, data_line)
                    )

                    logger_write(err_msg, logfile=self.logger)

                for item in values:

                    # Check if the ID= field exists
                    if first_item:
                        first_item = False

                        if item[0] == "ID":
                            snt_id = item[1]
                            id_field = True

                            # Check if the sentence ID is unique
                            if snt_id not in data_dict.keys():
                                data_dict[snt_id] = {}
                            else:
                                err_msg = (
                                    'the ID=%s in the file %s is not '
                                    'unique. There must be a different ID for '
                                    'each different sentence'
                                    % (snt_id, self.scp)
                                )

                                logger_write(err_msg, logfile=self.logger)

                            continue

                        else:

                            err_msg = (
                                'the first field of the %s must be "ID=" '
                                '(got "%s")'
                                % (self.scp, item)
                            )

                            logger_write(err_msg, logfile=self.logger)

                    # Check if the duration= field exists
                    if item[0] == "duration":

                        # Check if duration is a number
                        try:
                            duration = float(item[1])
                            total_duration = total_duration + duration
                        except Exception:

                            err_msg = (
                                'the "duration" specified in the line %s '
                                'of the scp file %s must be a float (got %s).'
                                % (data_line, self.scp, item[1])
                            )

                            logger_write(err_msg, logfile=self.logger)

                        data_dict[snt_id]["duration"] = duration

                        duration_field = True
                        continue

                    if id_field is True and duration_field is True:
                        data_name = item[0]
                        data_file_format = extr_brakets.findall(item[1])[0]

                        if len(data_file_format) == 0:

                            err_msg = (
                                'the field "%s" of the line "%s" of the '
                                'scp file "%s" must be formatted as '
                                '%s=(file,format) '
                                '(got "%s", missing bracket).'
                                % (
                                    data_name,
                                    data_line,
                                    self.scp,
                                    data_name,
                                    item[1],
                                )
                            )

                            logger_write(err_msg, logfile=self.logger)

                        data_file_format = data_file_format.split(",")

                        if len(data_file_format) < 2:

                            err_msg = (
                                'the field "%s" of the line "%s" of the '
                                'scp file "%s" must be formatted as '
                                "%s=(file,format)"
                                '(got "%s", missing comma).'
                                % (
                                    data_name,
                                    data_line,
                                    self.scp,
                                    data_name,
                                    item[1],
                                )
                            )

                            logger_write(err_msg, logfile=self.logger)

                        data_file = data_file_format[0]
                        data_format = data_file_format[1]

                        # Reading the additional options for the data reader
                        data_options = {}
                        if len(data_file_format) > 1:
                            opt_lst = data_file_format[2:]

                            for opt in opt_lst:
                                if ":" not in opt:

                                    err_msg = (
                                        'the field "%s" of the line "%s" '
                                        'of the scp file "%s" must be'
                                        'formatted as %s=(file,format,'
                                        'option1:value1,option2: value2,..)'
                                        '(got "%s", missing ":" for '
                                        'the additional options).'
                                        % (
                                            data_name,
                                            data_line,
                                            self.scp,
                                            data_name,
                                            item[1],
                                        )
                                    )

                                    logger_write(err_msg, logfile=self.logger)

                                opt_name = opt.split(":")[0]
                                opt_value = opt.split(":")[1]

                                # Saving the additional options in a dictionary
                                data_options[opt_name] = opt_value

                        # Check if the format specified is supported
                        if data_format not in self.supported_formats:

                            err_msg = (
                                'the format "%s" specified in line "%s" of '
                                'the scp file %s is not supported. The toolkit'
                                ' only supports the following formats %s.'
                                % (
                                    data_format,
                                    data_line,
                                    self.scp,
                                    self.supported_formats.keys(),
                                )
                            )

                            logger_write(err_msg, logfile=self.logger)

                        if self.supported_formats[data_format]["file"]:

                            # if needed, expand the filename with the
                            # specified folders (e.g, $data_folder will be
                            # replaced with the path specified in the "folders"
                            #  path)
                            if self.global_config is not None:
                                for folder in self.global_config.keys():
                                    data_file = data_file.replace(
                                        "$" + folder,
                                        self.global_config[folder],
                                    )

                            # Check if file exists
                            if not os.path.exists(data_file):

                                err_msg = (
                                    'the file %s specified in the file %s '
                                    'does not exist"'
                                    % (data_file, self.scp)
                                )

                                logger_write(err_msg, logfile=self.logger)

                        # If everything is fine let's add this entry into the
                        # data_dictionary:
                        data_dict[snt_id][data_name] = {}
                        data_dict[snt_id][data_name]["data"] = data_file
                        data_dict[snt_id][data_name]["format"] = data_format
                        data_dict[snt_id][data_name]["options"] = data_options

                    # Check if the ID field exits
                    if not id_field:

                        err_msg = (
                            'the line "%s" of the file %s does not contain '
                            'the required field "ID="'
                            % (data_line, self.scp)
                        )

                        logger_write(err_msg, logfile=self.logger)

                    # check if the duration filed exits
                    if not duration_field:

                        err_msg = (
                            'the line "%s" of the file %s does not contain '
                            'the required field "duration="'
                            % (data_line, self.scp)
                        )

                        logger_write(err_msg, logfile=self.logger)

                # If needed, select a subset of the data as specified in \
                # self_n_lines
                data_count = data_count + 1
                if self.select_n_sentences is not None:
                    if data_count == self.select_n_sentences:
                        break

                if data_count == 1:
                    data_entries = list(data_dict[snt_id].keys())
                else:
                    current_entries = list(data_dict[snt_id].keys())
                    if current_entries != data_entries:

                        err_msg = (
                            'the line "%s" of the file %s contains data '
                            'entries different from the previous lines '
                            '(got %s, expected %s)"'
                            % (
                                data_line,
                                self.scp,
                                current_entries,
                                data_entries,
                            )
                        )

                        logger_write(err_msg, logfile=self.logger)

        # Sorting the sentences as specified
        data_dict = self.sort_sentences(data_dict, self.sentence_sorting)

        log_text = (
            "\tNumber of sentences: %i\n" % (len(data_dict.keys()))
            + "\tTotal duration (hours): %1.2f \n" % (total_duration / 3600)
            + "\tAverage duration (seconds): %1.2f \n"
            % (total_duration / len(data_dict.keys()))
        )

        logger_write(log_text, logfile=self.logger, level="debug")

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

         Input:       - self (type:  create_dataloader class, mandatory)

                      - data_dict (type: dict, mandatory):
                       it is a dictionary with the data itemized in the scp
                       file.

                      - sorting ('ascending','descending','random','original',
                        mandatory):
                           it is a dictionary with the data itemized in the
                           scp file.

         Output:    - sorted_dictionary (type: dict):
                       it is a dictionary with the sorted data

         Example:  from data_io import create_dataloader

                   config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

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
                key=lambda k: data_dict[k]["duration"],
            )

        # Descending sorting
        if sorting == "descending":
            sorted_ids = sorted(
                sorted(data_dict.keys()),
                key=lambda k: -data_dict[k]["duration"],
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

         Input:       - self (type:  create_dataset class, mandatory)


         Output:    - supported_formats (type: dict):
                       it is a dictionary contained the supported formats and
                       the related readers.

         Example:  from data_io import create_dataloader

                   config={'class_name':'core.loop',\
                           'scp':'samples/audio_samples/scp_example2.scp'}

                   # Initialization of the class
                   data_loader=create_dataloader(config)

                   # Getting the supported reading formats
                   print(data_loader.get_supported_formats())
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
                       scp file.

                    - supported_formats (type: dict, mandatory):
                       it is a dictionary contaning the reading supported
                       format.

                   - data_entry (type: list, mandatory):
                       it is a list containing the data_entries to read from
                       the scp file.

                    - do_cache(bool,optional,Default:False):
                       When set to true, this option stores the input data
                       in a variable called self.cache. In practice, the
                       first time the data are read from the disk, they are
                       stored in the cpu RAM. If the data needs to be used
                       again (e.g. when loops>1) the data will be read
                       from the RAM directly. If False, data are read from
                       the disk every time.  Data are stored until a
                       certain percentage of the total ram available is
                       reached (see cache_ram_percent below)

                   - cache_ram_percent (int(0,100),optional,default:75):
                     If cache if True, data will be stored in the cpu
                     RAM until the total RAM occupation is less or equal
                     than the specified threshold. In practice, if a lot
                     of RAM is available several  data will be stored in
                     memory, otherwise, most of them will be read from the
                     disk directly.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.



     Input (call,__getitem__):  - idx (type: int):
                                   it is the index to read from the data list
                                   stored in data_dict['data_list'].

     Output (call,__getitem__):  - data: (type: list):
                                   it is a list containing the data. The list
                                   is formatted in the following way:
                                   [data_id,data,data_len]

     Example:  from data_io import create_dataloader
               from data_io import create_dataset

               config={'class_name':'core.loop',\
                       'scp':'samples/audio_samples/scp_example2.scp'}

               # Initialization of the data_loader class
               data_loader=create_dataloader(config)

               # data_dict creation
               data_dict=data_loader.generate_data_dict()

               # supported formats
               formats=data_loader.get_supported_formats()

               # Initialization of the dataser class
               dataset=create_dataset(data_dict,formats,'wav',False,0)

              # Reading data
              print(dataset.__getitem__(0))
              print(dataset.__getitem__(1))
     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        data_dict,
        supported_formats,
        data_entry,
        do_cache,
        cache_ram_percent,
        logger=None,
    ):

        # Setting the variables
        self.data_dict = data_dict
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

         Input:       - self (type:  create_dataset class, mandatory)


         Output:    - data_len (type: int):
                      it is the number of data to read (i.e. len of the
                      data_list entry of the data_dict).

         Example:  from data_io import create_dataloader
                   from data_io import create_dataset

                   config={'class_name':'core.loop',\
                           'scp':'samples/audio_samples/scp_example2.scp'}

                   # Initialization of the data_loader class
                   data_loader=create_dataloader(config)

                   # data_dict creation
                   data_dict=data_loader.generate_data_dict()

                   # supported formats
                   formats=data_loader.get_supported_formats()

                   # Initialization of the dataser class
                   dataset=create_dataset(data_dict,formats,'wav',False,0)

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

        # Managing caching
        if self.do_cache:

            if snt_id not in self.cache:

                # Reading data
                data = self.read_data(data_line, snt_id)

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
            data = self.read_data(data_line, snt_id)

        return data

    def read_data(self, data_line, snt_id):
        """
         ---------------------------------------------------------------------
         data_io.create_dataloader.read_data (author: Mirco Ravanelli)

         Description: This function manages reading operation from disk.

         Input:       - self (type:  create_dataset class, mandatory)

                       - data_line (type: dict, mandatory):
                           it is one of entries extreacted from the data_dict.
                           It contains all the needed information to read the
                           data from the disk.

                       - snt_id (type: str, mandatory):
                           it the sentence identifier.

         Output:    - data_read (type: list):
                      it is a list contaning the read data. The list if
                      formatted in the followig way: [data_id,data_data_len]

         Example:  from data_io import create_dataloader
                   from data_io import create_dataset

                   config={'class_name':'core.loop',\
                           'scp':'samples/audio_samples/scp_example2.scp'}

                   # Initialization of the data_loader class
                   data_loader=create_dataloader(config)

                   # data_dict creation
                   data_dict=data_loader.generate_data_dict()

                   # supported formats
                   formats=data_loader.get_supported_formats()

                   # Initialization of the dataser class
                   dataset=create_dataset(data_dict,formats,'wav',False,0)

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
            data_source, data_options=data_options, logger=self.logger
        )

        # Convert numpy array to float32
        if isinstance(data, np.ndarray):
            data_shape = np.asarray(data.shape[-1]).astype("float32")
        else:
            data_shape = np.asarray(1).astype("float32")

        data_read = [snt_id, data, data_shape]

        return data_read


def read_wav_soundfile(file, data_options={}, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.read_wav_soundfile (author: Mirco Ravanelli)

     Description: This function reads audio files with soundfile.

     Input (call):     - file (type: file, mandatory):
                           it is the file to read.

                       - data_options(type: dict, mandatory):
                           it is a dictionary containing options for the
                           reader.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  signal (type: numpy.array):
                       it is the array containing the read signal


     Example:  from data_io import read_wav_soundfile

               print(read_wav_soundfile('samples/audio_samples/example1.wav'))

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

            logger_write(err_msg, logfile=logger)

    # Managing start option
    if "start" in data_options:
        try:
            start = int(data_options["start"])
        except Exception:

            err_msg = (
                'The start value for the file %s must be an integer '
                '(e.g start:405)'
                % (file)
            )

            logger_write(err_msg, logfile=logger)

    # Managing stop option
    if "stop" in data_options:
        try:
            stop = int(data_options["stop"])
        except Exception:

            err_msg = (
                'The stop value for the file %s must be an integer '
                '(e.g stop:405)'
                % (file)
            )

            logger_write(err_msg, logfile=logger)

    # Managing samplerate option
    if "samplerate" in data_options:
        try:
            samplerate = int(data_options["samplerate"])
        except Exception:

            err_msg = (
                'The sampling rate for the file %s must be an integer '
                '(e.g samplingrate:16000)'
                % (file)
            )

            logger_write(err_msg, logfile=logger)

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
                'The number of channels for the file %s must be an integer '
                '(e.g channels:2)'
                % (file)
            )

            logger_write(err_msg, logfile=logger)

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
        logger_write(err_msg, logfile=logger)

    # Set time_steps always last as last dimension
    if len(signal.shape) > 1:
        signal = signal.transpose()

    return signal


def read_pkl(file, data_options={}, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.read_pkl (author: Mirco Ravanelli)

     Description: This function reads tensors store in pkl format.

     Input (call):     - file (type: file, mandatory):
                           it is the file to read.

                       - data_options(type: dict, mandatory):
                           it is a dictionary containing options for the
                           reader.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  tensor (type: numpy.array):
                       it is the array containing the read signal.


     Example:  from data_io import read_pkl

               print(read_pkl('pkl_file.pkl'))

     -------------------------------------------------------------------------
     """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            tensor = pickle.load(f)
    except Exception:
        err_msg = "cannot read the pkl file %s" % (file)
        logger_write(err_msg, logfile=logger)

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int32")

    return tensor


def read_string(string, data_options={}, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.read_string (author: Mirco Ravanelli)

     Description: This function reads data in string format.

     Input (call):     - string (type: str, mandatory):
                           it is the string to read

                       - data_options(type: dict, mandatory):
                           it is a dictionary containing options for the
                           reader.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  tensor (type: numpy.array):
                       it is the array containing the read signal.


     Example:  from data_io import read_string

               print(read_string('hello_world'))

     -------------------------------------------------------------------------
     """

    # Splitting elements with '_'
    string = string.split("_")
    return string


def read_kaldi_lab(kaldi_ali, kaldi_lab_opts, logfile=None):
    """
     -------------------------------------------------------------------------
     data_io.read_kaldi_lab (author: Mirco Ravanelli)

     Description: This function reads label in kaldi format

     Input (call):     - kaldi_ali (type: directory, mandatory):
                           it is the directory where kaldi alignents are
                           stored.

                       - kaldi_lab_opts(type: str, mandatory):
                           it is a string that contains the options for
                           reading the kaldi alignments.

                      - logfile (type: logger, optional, default: None):
                          it the logger used to write debug and error msgs.


     Output (call):  lab (type: dict):
                       it is a dictionary contaning the labels


     Example:  from data_io import read_kaldi_lab

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

     Input (call):     - data (type: torch.Tensor, mandatory):
                           it is the tensor to store as and audio file

                       - filename (type: file, mandatory):
                         it is the file where writign the data.

                      - sampling_rate (type: int, optional, default: None):
                       it sampling rate of the audio file.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from data_io import write_wav_soundfile

               signal=0.1*torch.rand([16000])
               write_wav_soundfile(signal,'exp/wav_example.wav',
               sampling_rate=16000)

     -------------------------------------------------------------------------
     """

    # Switch from (channel,time) to (time,channel) as Expected
    if len(data.shape) > 2:

        err_msg = (
            'expected signal in the format (channel,time). Got %s '
            'dimensions instead of two for file %s'
            % (len(data.shape), filename)
        )

        logger_write(err_msg, logfile=logger)

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
        logger_write(err_msg, logfile=logger)


def write_txt_file(data, filename, sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.write_txt_file (author: Mirco Ravanelli)

     Description: This function write data in text format

     Input (call):     - data (type: torch.Tensor, np.ndarray, str, list,
                         mandatory):
                           it is the data to write in the text file

                       - filename (type: file, mandatory):
                         it is the file where writing the data.

                      - sampling_rate (type: int, optional, default: None):
                       it sampling rate of the audio file.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from data_io import write_txt_file

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
            logger_write(err_msg, logfile=logger)

    # Opening the file
    try:
        file_id = open(filename, "w")

    except Exception:
        err_msg = "cannot create the file %s." % (filename)
        logger_write(err_msg, logfile=logger)

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

     Input (call):    - data (type: torch.Tensor, np.ndarray, str, list,
                         mandatory):
                           it is the data to write in the text file

                      - filename (type: file, mandatory):
                         it is the file where writing the data.

                      - sampling_rate (type: int, optional, default: None):
                       it sampling rate of the audio file.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  None


     Example:  import torch
               from data_io import write_stdout

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

     Input (call):     - data (type: torch.Tensor, mandatory):
                           it is the tensor to write in the text file

                       - filename (type: file, mandatory):
                         it is the file where writing the data.

                      - sampling_rate (type: int, optional, default: None):
                       it sampling rate of the audio file.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


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

        err_msg = (
                'cannot import matplotlib. Make sure it is installed.'
                % (filename, str(data.shape))
        )

        logger_write(err_msg, logfile=logger)
        raise

    # Checking tensor dimensionality
    if len(data.shape) < 2 or len(data.shape) > 3:

        err_msg = (
            'cannot save image  %s. Save in png format supports 2-D or '
            '3-D tensors only (e.g. [x,y] or [channel,x,y]). '
            'Got %s'
            % (filename, str(data.shape))
        )

        logger_write(err_msg, logfile=logger)

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
            logger_write(err_msg, logfile=logger)


def write_ark(data, filename, key="", sampling_rate=None, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.write_ark (author: github.com/vesis84/kaldi-io-for-python)

     Description: write_mat(data,filename key='')
                  Write a binary kaldi matrix to filename or stream.
                  Supports 32bit and 64bit floats.

     Input (call): - filename (type:file, mandatory):
                       it is filename of opened file descriptor for writing.

                   - data (type:np.ndarray, mandatory):
                       it is the matrix to be stored

                   - key (type:str, optional, default=''):
                       it is used for writing ark-file, the utterance-id gets
                       written before the matrix.

                   - sampling_rate (type: int, optional, default: None):
                       it sampling rate of the audio file.

                   - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.

     Output (call):  None

     -------------------------------------------------------------------------
     """

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    fd = open_or_fd(filename, None, mode="wb")
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

     Input (call):     - file (type: file, mandatory):
                           it is the file from which we want to computed the
                           md5


     Output (call):  md5 (type: md5):
                       it is the checksum for the given file


     Example:  from data_io import  get_md5
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

     Input (call):     - files (type: file_list, mandatory):
                           it is a list of input files from which we will
                           compute the md5.

                       - outfile (type: file, mandatory):
                           it is the path where storing the output pkl file.

     Output (call):  None


     Example:  from data_io import save_md5

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
     data_io.save_pkl (author: Mirco Ravanelli)

     Description: This function saves an object in pkl format.

     Input (call): - obj (type:obj, mandatory):
                       it object to save in pkl format

                   - file (type:file, mandatory):
                       it is name of the output file.

                   - sampling_rate (type: int, optional, default: None):
                       it sampling rate of the audio file.

                   - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.

     Output (call):  None


     Example:  from data_io import save_pkl

               out_file='exp/example.pkl'
               save_pkl([1,2,3,4,5],out_file)

     -------------------------------------------------------------------------
     """
    try:
        with open(file, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        err_msg = "Cannot save file %s" % (file)
        logger_write(err_msg, logfile=logger)


def load_pkl(file, logger=None):
    """
     -------------------------------------------------------------------------
     data_io.load_pkl (author: Mirco Ravanelli)

     Description: This function load a pkl file

     Input (call): - file (type:file, mandatory):
                       it is name of the input pkl file.

                   - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.

     Output (call):  obj (type:obj)


     Example:  from data_io import load_pkl

               pkl_file='exp/example.pkl'
               print(load_pkl(pkl_file))

     -------------------------------------------------------------------------
     """

    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except Exception:
        err_msg = "Cannot read file %s" % (file)
        logger_write(err_msg, logfile=logger)


def read_vec_int_ark(file_or_fd, logfile=None):
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


def read_vec_int(file_or_fd, logfile=None):
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


def open_or_fd(file, logfile=None, mode="rb"):
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
            "^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:", file
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


def popen(cmd, logfile=None, mode="rb"):
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
