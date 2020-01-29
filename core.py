"""
 -----------------------------------------------------------------------------
 core.py

 Description: This library gathers important classes that implement crucial
              functionalities of SpeechBrain.  All the classes are designed
              with the same input/output arguments such that they can be
              called within configuration files.
 -----------------------------------------------------------------------------
"""

# Importing libraries
import os
import sys
import torch
import itertools
import torch.multiprocessing as mp
from data_io import create_dataloader
from utils import (
    check_opts,
    import_class,
    logger_write,
    read_config,
    setup_logger,
    check_inputs,
)


class execute_computations:
    """
     -------------------------------------------------------------------------
     core.execute_computations (author: Mirco Ravanelli)

     Description: This class executes the computations reported in the config
                  file.

     Input (init):  - exec_config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - config_file (type: file, mandatory):
                               the config file that contains the computations
                               to execute.

                           - cfg_change (type:str,optional,default: None):
                               it can be used to change the param of the
                               cfg_file (e.g, cfg_change=--global,device=cuda)

                           - stop_at (type: str_list,optional, default: None):
                               it is used to stop the computations when the
                               variables or functions reported in stop_at are
                               encountered in the computation section.

                           - out_var (type: str_list,optional, default: None):
                               it is used to define the variables of
                               computation section that will be returned
                               when calling execute_computation.

                           - root_cfg (type: bool,optional, default: False):
                               it is a flag that indicates if the current
                               config file is the root one or not.

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


     Output (call):  - out_var_lst (type: list):
                       it returns the output variables defined in out_var in a
                       list. If out_var is None, a list containing [None] is
                       returned.

     Example:    from core import execute_computations

                 cfg_file='cfg/minimal_examples/data_reading/\
                 read_data_example.cfg'

                 # Definition of the exec_config dictionary
                 exec_config={'class_name':'core.execute_computations', \
                              'cfg_file': cfg_file, \
                              'root_cfg':'True'}

                # Initialization of the class
                computations=execute_computations(exec_config)

                # Executing computations
                computations([])
     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        exec_config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):

        # Setting logger and exec_config
        self.logger = logger
        self.exec_config = exec_config

        # Definition of the expected options
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "cfg_file": ("path", "mandatory"),
            "cfg_change": ("str", "optional", "None"),
            "stop_at": ("str_list", "optional", "None"),
            "out_var": ("str_list", "optional", "None"),
            "root_cfg": ("bool", "optional", "False"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, self.exec_config, self.logger
        )

        # Setting global config
        if global_config is None:
            self.global_config = {}
        else:
            self.global_config = global_config

        # Reading the config file to execute
        self.config = read_config(
            self.cfg_file,
            cfg_change=self.cfg_change,
            global_config=self.global_config,
            root_cfg=self.root_cfg,
        )

        # Setting other variables
        self.computation_dict = self.config["computations"]
        self.variables = {}

        # Declaring global function that gather all the functions
        # seen in all the config files
        if self.root_cfg:
            global functions
            functions = {}

        # Adding torch in input (to support commands like torch.cat, etc.)
        if "torch" in globals():
            self.variables["torch"] = globals()["torch"]

        # Supporting sys in pycmd (useful for stop the execution with sys.exit)
        if "sys" in globals():
            self.variables["sys"] = globals()["sys"]

        # Creating logger and output directory (if needed)
        if self.logger is None and self.root_cfg:

            # Output folder creation
            self.output_folder = self.config["global"]["output_folder"]
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            # Logger initialization
            log_file = self.config["global"]["output_folder"] + "/log.log"

            self.logger = setup_logger(
                "logger",
                log_file,
                verbosity_stdout=self.config["global"]["verbosity"],
            )

        # Set verbosity
        if "verbose" in self.config["global"]:
            self.verbose = self.config["global"]["verbose"]
        else:
            self.verbose = True

        # Scanning the computation to decide at which computation
        # the needed variables in self.stop_at can be returned.
        stop_computations = self.detect_stop_computations(self.stop_at)

        # Check at which computation the output variables can be returned
        last_out_var_computation = self.detect_stop_computations(self.out_var)

        # Stopping the computation when encoutering the variables in
        # self.stop_at. If the user requires a variable that comes after
        # that, by default we perform the computation until all the out_var
        # are founded (this is implemented by the following max).

        self.stop_computations = max(stop_computations,
                                     last_out_var_computation)

        # if both stop_at and out_var are None, we will do all the computations
        if self.stop_computations == -1:
            self.stop_computations = len(self.computation_dict)-1

        # Do basic prints only if verbose and root_cfg is True
        self.add_prints = logger is not None and self.verbose and self.root_cfg

        # Basic prints
        if self.add_prints:
            text = (
                "-------------------------------------------------------\n"
                + "Running computations...\n"
                + "-------------------------------------------------------\n\n"
                + "config_file = "
                + self.cfg_file
                + "\n"
            )

            logger_write(text, logfile=self.logger, level="info")

    def __call__(self, inp_lst):

        # Reading the input list
        self.input_var = inp_lst

        # Executing the computations the computation_dict (line by line)
        for comp_id in self.computation_dict:

            # Reading a block of computations
            computation_list = self.computation_dict[comp_id]

            # Parallel processing (multiple computations to run in parallel)
            if len(computation_list) > 1:

                # Running computations on different processes
                jobs = []

                if self.add_prints:
                    funct_names = []

                    for computation in computation_list:
                        funct_names.append(computation["funct_name"])

                    text = "- STEP %i: running %s (in parallel)...\n" % (
                        comp_id,
                        " and ".join(funct_names),
                    )
                    logger_write(text, logfile=self.logger, level="info")

                # Initializing the local_variables (it must be a shared dict
                # with multiple processes)
                manager = mp.Manager()
                local_variables = manager.dict()

                # Doing computations of different processes
                for computation in computation_list:
                    p = mp.Process(
                        target=self.run_function,
                        args=([computation]),
                        kwargs={"local_variables": local_variables},
                    )
                    jobs.append(p)
                    p.start()

                # Waiting for all the processes
                for j in jobs:
                    j.join()

                # Update variables with the local one
                self.variables.update(local_variables)

                if self.add_prints:
                    text = "\tSTEP %i COMPLETED SUCCESSFULLY!\n" % (comp_id)
                    logger_write(text, logfile=self.logger, level="info")

            # Single computation
            else:
                if self.add_prints:
                    text = '- STEP %i: running "%s"...\n' % (
                        comp_id,
                        computation_list[0]["funct_name"],
                    )
                    logger_write(text, logfile=self.logger, level="info")

                loc_var = functions

                # Running the computation
                self.run_function(computation_list[0], local_variables=loc_var)

                # Update the variable dictionary
                self.variables.update(loc_var)

                # Adding prints
                if self.add_prints:
                    text = "\tSTEP %i COMPLETED SUCCESSFULLY!\n" % (comp_id)
                    logger_write(text, logfile=self.logger, level="info")

            # Stop the computation when all output_variabels are defined
            if comp_id == self.stop_computations:

                out_var_lst = []

                if self.out_var is not None:

                    # Return in output a list as specified in out_var
                    out_var_lst = []

                    for var in self.out_var:
                        out_var_lst.append(self.variables[var])

                else:
                    # Return None if out_var is not specified
                    out_var_lst = [None]

                # Final prints
                if self.add_prints:
                    text = "Computations COMPLETED SUCCESSFULLY!\n"
                    logger_write(text, logfile=self.logger, level="info")

                return out_var_lst

    def run_function(self, computation, local_variables={}):
        """
         ---------------------------------------------------------------------
         core.execute_computations.run_function (author: Mirco Ravanelli)

         Description: This function executes the given computation.The
                      computation must contain one of the functions declared
                      in the [functions] section of the config file. The input
                      and the output of each computation is a list of
                      variables. The output variables are added into the
                      self.variables that gathers all the variables defined in
                      all the [computation] section of the config file.
                      Run_function also supports two special functions called
                      get_inp_var() and pycmd. get_inp_var() simply return the
                      input variables given when  calling the
                      execute_computation class, while pycmd can be used to
                      run python commands.

         Input:        - self (type, execute_computaion class, mandatory)

                       - computation (type, dict, mandatory):
                           it is a dictionary that describes the computation
                           to be executed

                       - local_variables (type, dict, optional,
                         default: None):
                           it is a dictionary that will be updated with the
                           local.

         Output:     None

         Example:    from core import execute_computations

                     cfg='cfg/minimal_examples/data_reading/\
                     read_data_example.cfg'

                     # Definition of the exec_config dictionary
                     exec_config={'class_name':'core.execute_computations', \
                                  'cfg_file': cfg, \
                                  'root_cfg':'True'}

                    # Initialization of the class
                    computations=execute_computations(exec_config)

                    # Running the first computation:
                    computation=computations.computation_dict[0][0]
                    computations.run_function(computation)
         ---------------------------------------------------------------------
         """

        # Putting input variables in input list
        input_lst = []
        for inp in computation["inputs"]:
            input_lst.append(self.variables[inp])

        try:

            funct_name = computation["funct_name"]

            # Initializing of the class (if called for the first time)
            if funct_name not in functions:

                if funct_name in self.config["functions"]:
                    library = self.config["functions"][funct_name][
                        "class_name"
                    ]

                    # Importing Library
                    try:
                        functions[funct_name] = import_class(library)
                    except Exception:
                        err_msg = "Cannot import class %s" % (library)
                        logger_write(
                            err_msg, logfile=self.logger, level="debug"
                        )
                        raise

                    # Function Initialization
                    try:
                        functions[funct_name] = functions[
                            funct_name
                        ](
                            self.config["functions"][funct_name],
                            funct_name=funct_name,
                            global_config=self.global_config,
                            logger=self.logger,
                            first_input=input_lst,
                        )
                    except Exception:
                        err_msg = "Cannot initialize function %s" % (
                            funct_name
                        )
                        logger_write(
                            err_msg, logfile=self.logger, level="debug"
                        )
                        raise

                else:
                    # Raising an error in the function is not defined
                    if funct_name != "pycmd" and funct_name != "get_input_var":
                        err_msg = (
                            'the function "%s" reported in the [computations] '
                            'section of the config file %s is not defined'
                            % (funct_name, self.exec_config["cfg_file"])
                        )

                        logger_write(err_msg)

            # Executing the function
            if funct_name != "pycmd":

                # Managing the special function get_input_var()
                if funct_name == "get_input_var":
                    result = self.input_var
                else:
                    # Function execution
                    result = functions[funct_name](input_lst)

                if result is None:
                    result = []

                # Check if the number of output is the one expected
                if len(computation["outputs"]) > len(result):
                    err_msg = (
                        'the function "%s" returns %i arguments, while in '
                        'the line %s of the [computations] section of the '
                        'config file %s %i arguments are expected.'
                        % (
                            funct_name,
                            len(result),
                            computation["raw_line"],
                            self.exec_config["cfg_file"],
                            len(computation["outputs"]),
                        )
                    )

                    logger_write(err_msg, logfile=self.logger)

                # Saving all returned local variables
                for i in range(len(computation["outputs"])):
                    out_name = computation["outputs"][i]

                    # Adding results to local variables
                    try:
                        local_variables[out_name] = result[i]
                    except Exception:
                        err_msg = (
                            "Cannot execute self.variables[out_name]=result[i]"
                        )
                        logger_write(
                            err_msg, logfile=self.logger, level="debug"
                        )
                        raise

                    # Adding current class to local variable
                    if funct_name != 'get_input_var':

                        local_variables[funct_name] = \
                            functions[funct_name]

            # Managing the special function pycmd (that runs python commands)
            else:
                try:
                    cmd = computation["pycmd"][0]

                    # Executing the command
                    exec(cmd, self.variables, local_variables)

                except Exception:
                    err_msg = (
                        'Cannot execute the python command "%s" defined in '
                        'the [computations] section of the config file %s'
                        % (
                            computation["raw_line"],
                            self.exec_config["cfg_file"],
                        )
                    )

                    logger_write(err_msg, logfile=self.logger, level="debug")
                    raise

        # Raising an error when running a parallel process on the same GPU
        except RuntimeError as e:
            if "CUDA error: initialization error" in str(e):
                err_msg = ('The current version does not support parallel'
                           'computations on the same gpu!!')
                logger_write(err_msg, logfile=self.logger, level="debug")
                raise
            else:
                err_msg = (
                    'Cannot run the function "%s" defined in the '
                    '[computations] section of the config file %s'
                    % (funct_name, self.exec_config["cfg_file"])
                )

                logger_write(err_msg, logfile=self.logger, level="debug")
                raise

    def detect_stop_computations(self, stop_var):
        """
         ---------------------------------------------------------------------
         core.execute_computations.detect_stop_computations
         (author: Mirco Ravanelli)

         Description: This function analyzes the computation dictionary and
                      variables in stop_var. It returns the
                      computation_id where we can stop the computations (i.e,
                      the last time the variables are encountered in the
                      computation section).

         Input:       - self (type: execute_computaion class, mandatory)
                      - stop_var (type: list, mandatory):
                          it is a list of variables where to stop the
                          computations.


         Output:      - stop_id (type: str)

         Example:    from core import execute_computations

                     cfg_file='cfg/minimal_examples/features/MFCCs.cfg'

                     # Definition of the exec_config dictionary
                     exec_config={'class_name':'core.execute_computations', \
                                  'cfg_file': cfg_file}

                    # Initialization of the class
                    computations=execute_computations(exec_config)

                    # Running detect_stop_computations()
                    stop_id=computations.detect_stop_computations(
                    ['STFT','MFCCs'])

                    # print the results
                    print(computations.computation_dict.keys())
                    print(stop_id)
         ---------------------------------------------------------------------
         """

        # If stop_var is None, we will return -1 as computation_id
        stop_computations = -1

        if stop_var is not None and len(stop_var) > 0:

            found_var = []
            stop_computations = []

            for block in self.computation_dict:
                for computation in self.computation_dict[block]:
                    for var in stop_var:

                        if var in computation["outputs"] or \
                           var in computation["funct_name"]:

                            found_var.append(var)

                            if set(found_var) == set(stop_var):
                                # return the computation-id where stopping
                                # computations
                                stop_computations.append(block)

            # Check if the all the out variables are encountered
            if len(stop_computations) == 0:
                err_msg = (
                    'the output variables (stop_var=%s) defined in '
                    'execute_computations are not defined in the section  '
                    '[computation] of the config file!'
                    % (stop_var)
                )

                logger_write(err_msg)

            # Returning the last computation where the variable is assigned
            stop_computations = max(stop_computations)

        return stop_computations


class loop:
    """
     -------------------------------------------------------------------------
     core.loop (author: Mirco Ravanelli)

     Description: This class loops over the data reported in the scp file.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - scp (type: file, optional, None):
                               it is the scp file that contains the data.
                               if not set, we only repeat the computations
                               of the processing_cfg file n_loops times.

                            - processing_cfg (type: file, optional, default:
                              None):
                                this optional field might contain a config
                                file that will be used to process the input
                                data. If not specified, we will just loop over
                                the scp data without doing any additional
                                processing. The computation section of the
                                processing_cfg file can contain a special
                                function called get_inp_var() (see for instance
                                cfg/minimal_examples/features/STFT.cfg). This
                                function returns the batch list passed in
                                self.loop_computation(batch). The list is
                                compososed of the sentence_id followed by data,
                                data_len of the data reported in the scp file
                                and read with the scp variable (see description
                                later). Finally the list contain the batch_id,
                                the iteration_id, as well as the inputs
                                specified when calling the loop class.

                           - cfg_change (type:str,optional,default: None):
                               it can be used to change the param of the
                               processing cfg_file
                               (e.g, cfg_change=--global,device=cuda)

                           - stop_at (type: str_list,optional, default: None):
                               when the processing_cfg file is set, stop_at
                               stops the execution of the processing function
                               when the given variables or function names are
                               encountered (by default, we return the values
                               observed the last time the variable is assigned.
                               It can be useful when we have to run only a part
                               of the computations reported in the
                               processing_cfg.

                           - out_var (type: str_list, optional, default: None):
                               it is used to define the variables of
                               computation section that will be returned
                               when calling execute_computation.

                            - accum_type ('list','sum','average','last',
                                          optional, default: None)
                                this variable defines the way the out_var
                                are accumulated over the different iterations
                                of the loop.
                                If set to 'list', the output
                                variables will be accumulated into a list.
                                The list has the same number of elements of
                                out_var. Each element is composed of another
                                list that contains each returned ouput variable
                                (i.e, each element will contain a list of
                                n_loop*data_size elements).
                                If set to 'sum', the elements are summed up,
                                while if set to 'average' the elements are
                                averaged (this is useful, for instance, when
                                computing the loss at each iteration and I
                                want to return the average loss over all the
                                iterations).
                                If set to 'last', only the last returned
                                element is saved (this can be useful for
                                instance when we want to return the final
                                model at the end of the training loop).


                            - torch_no_grad (type:bool, optional, def:False):
                                If True, the computations will be performed
                                with the flag torch.no_grad() as required
                                in the test/validation modality.

                           - batch_size: (type:int(1,inf),optional,default:1):
                               the data itemized in the scp file are
                               automatically organized in batches. In the case
                               of variable size tensors, zero padding is
                               performed. When
                               batch_size=1, the data are simply processed one
                               by one without the creation of batches.


                           - sentence_sorting: ('ascending,descending,random,
                             original', optional, 'original'):
                               This parameter specifies how to sort the data
                               before the batch creation. Ascending and
                               descending values sort the data using the
                               "duration" field in the scp files.
                               Random sort the data randomly, while original
                               (the default option) keeps the original
                               sequence of data defined in the scp file. Note
                               that this option affects the batch creation.
                               If the data are sorted in ascending or
                               descending order the batches will approximately
                               have the same size and the need for zero
                               padding is minimized. Instead, if
                               sentence_sorting is set to random, the batches
                               might be composed of both short and long
                               sequences and several zeros might be added in
                               the batch. When possible, it is desirable to
                               sort the data. This way, we use more
                               efficiently the computational resources,
                               without wasting time on processing time steps
                               composed on zeros only. Note that is the data
                               are sorted in ascending/descending errors the
                               same batches will be created every time we want
                               to loop over the dataset, while if we set a
                               random order the batches will be different
                               every time we loop over the dataset.

                           - scp_read (type: str_list,optional,default:None):
                               this option can be used to read only some
                               data_entries of the scp file.When not specified
                               it automatically reads all the data entries.

                           - select_n_sentences (type: int(1,inf),optional,
                             None):
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
                               batch of data.
                               Please, see the pytorch documentation on the
                               data loader for more details.

                           - cache(bool,optional,Default:False):
                               When set to true, this option stores the input
                               data in a variable called self.cache (see
                               create_dataloader in data_io.py). In practice,
                               the first time the data are read from the disk,
                               they are stored in the cpu RAM. If the data
                               needs to be used again (e.g. when loops>1)
                               the data will be read from the RAM directly.
                               If False, data are read from the disk every
                               time.
                               Data are stored until a certain percentage of
                               the total ram available is reached
                               (see cache_ram_percent below).

                           - cache_ram_percent (int(0,100),optional,
                             default:75):
                               If cache if True, data will be stored in the
                               cpu RAM until the total RAM occupation is less
                               or equal than the specified threshold
                               (by default 75%). In practice, if a lot of RAM
                               is available several data will be stored
                               in memory, otherwise, most of them will be read
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


     Input (call): - inp_lst(type: list, mandatory):
                       by default the input arguments are passed with a list.


     Output (call):  out_var_lst (type: list):
                         it is a list containing all the
                         output variables accumulated over the n_loops.
                         The list depends on how out_var and accum_type are
                         set (please read above for more details on these
                         options).

     Example:   from core import loop

                config={'class_name':'core.loop',\
                         'scp':'samples/audio_samples/scp_example2.scp'}

                # Initialization of the class
                do_loop=loop(config)

                # Executing computations
                do_loop([])
     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):

        # Logger
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "scp": ("file", "optional", "None"),
            "processing_cfg": ("file", "optional", "None"),
            "cfg_change": ("str", "optional", "None"),
            "n_loops": ("int(1,inf)", "optional", "1"),
            "stop_at": ("str_list", "optional", "None"),
            "out_var": ("str_list", "optional", "None"),
            "accum_type": ("one_of_list(list,sum,average,last)",
                           "optional", "None"),
            "torch_no_grad": ("bool", "optional", "False"),
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
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, logger=self.logger
        )

        # Creating dataloader for the specified scp file
        if self.scp is not None:
            dataloader_obj = create_dataloader(
                config, global_config=global_config, logger=self.logger
            )
            self.dataloader = dataloader_obj.dataloader

            # Updating scp read
            self.scp_read = dataloader_obj.scp_read

        # Running the processing_cfg file is not None
        if self.processing_cfg is not None:

            # Make sure accum_type has the same length of out_var when defined
            if self.out_var is not None:

                if self.accum_type is not None:

                    if len(self.accum_type) != len(self.out_var):

                        err_msg = (
                            'the field accum_type must be a list with the '
                            'same  length  of the one specified in out_var.'
                            'Got %i vs %i'
                            % (len(self.accum_type), len(self.out_var))
                        )

                        logger_write(err_msg, logfile=logger)
                else:

                    self.accum_type = ['list'
                                       for i in range(len(self.out_var))]

            # Initializing the execute computation class
            exec_config = {"class_name": "core.execute_computations"}

            if self.processing_cfg is not None:
                exec_config.update({"cfg_file": self.processing_cfg})

            if self.cfg_change is not None:
                exec_config.update({"cfg_change": self.cfg_change})

            if self.stop_at is not None:
                exec_config.update({"stop_at": ",".join(self.stop_at)})

            if self.out_var is not None:
                exec_config.update({"out_var": ",".join(self.out_var)})

            # Running the processing computations
            self.loop_computation = execute_computations(
                exec_config, global_config=global_config, logger=self.logger
            )

    def __call__(self, inp):

        # Initializing the output_var list
        self.out_var_lst = []

        # Managing accumulation of the output
        if self.out_var is not None:

            for i, var in enumerate(self.out_var):

                if self.accum_type[i] == 'list' or\
                   self.accum_type[i] == 'last':

                    self.out_var_lst.append([])

                if self.accum_type[i] == 'sum' or \
                   self.accum_type[i] == 'average':

                    self.out_var_lst.append(0)
        else:
            self.out_var = []

        # Processing the data for n_loops
        for i in range(self.n_loops):

            # Managing data loops where we loop also over the data in the scp
            # file
            if self.scp is not None:

                # Calling the dataloader
                for batch_id, data_list in enumerate(zip(*self.dataloader)):

                    # Process data list to have a list formatted in this way
                    # [snt_id,data1,data1_len,data2,data2_len,..]
                    batch = self.prepare_data_list(data_list)

                    # Appending batch_id
                    batch.append(batch_id)

                    # Appending iteration_id
                    batch.append(i)

                    # Appending input list
                    batch = batch + inp

                    # Processing the batch as specified in self.processing_cfg
                    self.run_computations(batch)

                # Appending the last output to out_var_lst if needed
                if len(self.out_var) > 0:
                    for out_id, out in enumerate(self.output):

                        if self.accum_type[out_id] == 'last':
                            self.out_var_lst[out_id] = out

                        if self.accum_type[out_id] == 'average':
                            N_elements = (i+1)*(batch_id+1)

                            self.out_var_lst[out_id] =\
                                self.out_var_lst[out_id] / N_elements

            # Managing standard loop where we only replicate the computations
            # n_loops time (without looping also over the scp data)
            else:
                # Running the computations reported in self.processing_cfg
                self.run_computations([i])

        return self.out_var_lst

    def run_computations(self, inp):
        """
         ---------------------------------------------------------------------
         core.loop.run_computations (author: Mirco Ravanelli)

         Description: This function executes the computations specified
                      in the processing_cfg file. It also stored the
                      outputs, detaches the related tensors, and accumulates
                      it in the self.out_var list that will gather all the
                      outputs observed over the different iterations.

         Input:        - self (type, loop class, mandatory)
                       - inp (type, list, mandatory):
                           it is the input to give to the loop computation
                           function. It corresponds to what one can read with
                           the special function get_input_var that you can
                           call in the computation section of the config file.


         Output:      - data_list (type: list)

         Example:   from core import loop

                    config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

                    # Initialization of the class
                    do_loop=loop(config)

                    # executing computations
                    do_loop.out_var_lst = []
                    do_loop.run_computations([])

         ---------------------------------------------------------------------
         """

        # Running the computations reported in the processing_cfg file
        if self.processing_cfg is not None:

            if self.torch_no_grad:
                with torch.no_grad():
                    self.output = self.loop_computation(inp)

            else:
                self.output = self.loop_computation(inp)

        # Storing the output in a list if required
        if len(self.out_var) > 0:

            # Detatching torch.Tensors before accumulation
            self.output = self.detach_tensor_in_list(self.output)

            # Accumulating outputs as specified in accum_type
            for i, out in enumerate(self.output):

                # Appending out_var in a list
                if self.accum_type[i] == 'list':
                    self.out_var_lst[i].append(out)

                # Summing up the returned out_vars
                if self.accum_type[i] == 'sum' or \
                   self.accum_type[i] == 'average':

                    self.out_var_lst[i] = self.out_var_lst[i]+out

    @staticmethod
    def detach_tensor_in_list(lst):
        """
         ---------------------------------------------------------------------
         core.loop.detach_tensor_in_list (author: Mirco Ravanelli)

         Description: This scans a list and detaches the tensors in it

         Input:        - lst (type:list, mandatory):
                         it is a list whose tensors need to be detached.


         Output:      - out_lst (type: list):
                         it is the output list with the tensor detached

         Example:   import torch
                    from core import loop

                     config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

                    # Initialization of the class
                    do_loop=loop(config)

                    # Example of a list
                    data_list=[torch.Tensor([1,2,3,4]),'a','b']
                    # list conversion
                    data_list_out=do_loop.detach_tensor_in_list(data_list)

         ---------------------------------------------------------------------
         """

        # Initializing the output list
        lst_out = []

        # Looking for tensors and detaching them
        for elem in lst:
            if isinstance(elem, torch.Tensor):
                lst_out.append(elem.detach())
            else:
                lst_out.append(elem)

        return lst_out

    @staticmethod
    def prepare_data_list(data_list):
        """
         ---------------------------------------------------------------------
         core.loop.prepare_data_list (author: Mirco Ravanelli)

         Description: This function converts the data read by the
                      data_loader into a list formatted in this way:
                      [snt_ID, data, data_len, data2, data_len2,...]

         Input:       - data_list (type, list, mandatory)


         Output:      - data_list (type: list)

         Example:    from core import loop

                     config={'class_name':'core.loop',\
                             'scp':'samples/audio_samples/scp_example2.scp'}

                    # Initialization of the class
                    do_loop=loop(config)

                    # data_list (example of what returned by the dataloader)
                    data_list=[[['example1'],\
                                torch.tensor([1,2,3,4]),\
                                torch.tensor(1.0)], \
                                [['example1'],\
                                 [['spk01']],\
                                 torch.tensor(1.0)]]
                    # list conversion
                    data_list_out=do_loop.prepare_data_list(data_list)
                    print(data_list_out)

         ---------------------------------------------------------------------
         """

        # Saving sentence ids
        data_id = data_list[0][0]

        # Flatten the list
        data_list = list(itertools.chain(*data_list))

        # Remove multiple data_id entries
        del data_list[0::3]

        # Adding data_id to have a list like:
        # [snt_id,data1,data1_len,data2,data2_len,..]
        data_list.insert(0, data_id)

        return data_list
