"""
 -----------------------------------------------------------------------------
 core.py

 Description: This library gathers important classes that implement crucial
              functionalities of SpeechBrain.
 -----------------------------------------------------------------------------
"""

# Importing libraries
import os
import sys
import ast
import torch
import itertools
import torch.nn as nn
from tqdm import tqdm
from speechbrain.data_io.data_io import create_dataloader, load_pkl, save_pkl
from speechbrain.utils import (
    check_opts,
    import_class,
    logger_write,
    read_config,
    setup_logger,
    process_cmd_string,
    conf_to_text,
    write_config,
)


class execute_computations(nn.Module):
    """
     -------------------------------------------------------------------------
     core.execute_computations (author: Mirco Ravanelli)

     Description: This class executes the computations reported in the given
                  configuration file. When a csv file is specified, the
                  function automatically creates data batches and loops over
                  the data reported in the csv file. These data batches can be
                  processed by the config file reported in the cfg_file field.
                  The execute_computations class, actually implements a bunch
                  of functionalities that are useful for developing neural and
                  signal processing system efficiently (see the following
                  options)


     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.


                            - cfg_file (type: file, mandatory):
                                this field contains the path to the config
                                file that we want to execute. The computations
                                reported in the [computation] section will be
                                executed.

                           - cfg_change (type:str,optional,default: None):
                               it can be used to change the param of the
                               processing cfg_file
                               (e.g, cfg_change=--global,device=cuda)

                           - root_cfg (type:bool,optional,default: False):
                               it is a flag that tells wheter the current
                               config file is root (i.e., it is the first
                               one called) or not (i.e., it is called by
                               another config_file)). When calling the root_cfg
                               we initialize the output_folder and the logger.


                           - csv_file (type: file, optional, None):
                               it is the csv file that contains the data.
                               When specified, we automatically read the
                               file and create the needed batches of data.
                               When specified, the computations are executed
                               within a loop that loops over all the
                               data batches as well. One can use this option
                               when wants to process all the data batches
                               created from the csv_file.

                           - csv_read (type: str_list,optional,default:None):
                               this option can be used to read only some
                               data_entries of the csv file (e.g, wav and phn
                               only). When not specified it automatically
                               reads all the data entries.

                           - batch_size: (type:int(1,inf),optional,default:1):
                               the data itemized in the csv file are
                               automatically organized in batches. In the case
                               of variable size tensors, zero padding is
                               performed. When
                               batch_size=1, the data are simply processed one
                               by one without the creation of batches.


                           - stop_at (type: str_list,optional, default: None):
                               when the cfg_file file is set, stop_at
                               stops the execution of the processing function
                               when the given variables or function names are
                               encountered (by default, we return the values
                               observed the last time the variable is assigned.
                               It can be useful when we have to run only a part
                               of the computations reported in the
                               cfg_file.

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

                           - sentence_sorting: ('ascending,descending,random,
                             original', optional, 'original'):
                               This parameter specifies how to sort the data
                               before the batch creation. Ascending and
                               descending values sort the data using the
                               "duration" field in the csv files.
                               Random sort the data randomly, while original
                               (the default option) keeps the original
                               sequence of data defined in the csv file. Note
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


                           - select_n_sentences (type: int(1,inf),optional,
                             None):
                               this option can be used to read-only n
                               sentences from the csv file. This option can be
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

                            - n_loops (type:int(1,inf), optional, def:1):
                                it can be used to repeat the same computations
                                multiple times (e.g, when training a neural
                                network I have to repeat the same computations
                                N_epochs times).


                            -  replicate (type:int(1,inf), optional,def:1):
                                   this functionality can be used to pile up
                                   blocks of computations. For instance, in a
                                   standard neural network, we might have the
                                   same types of computations for N layers.
                                   Users can thus define the computations of
                                   the basic building block (e.g, a neural
                                   layer) and this function creates the final
                                   newtwork by replicating the computations of
                                   the basic block. The advantage of this is
                                   that we require users to only code the
                                   basic element and not the entire network
                                   every time. The other advantage is that
                                   when creating the final architecture we can
                                   automatically add residual, skip, or
                                   dense connections (without asking users to
                                   implement them for every model). The basic
                                   building block is replicated for the number
                                   of times reported in this field.
                                   For instance, if my basic building block is:

                                   out = linear(fea)
                                   out = relu(out)

                                   and I want to radd 3 layers, I have to
                                   set this parameter to 3. The computations
                                   that will be executed are:

                                   out = linear(fea)
                                   out = relu(out)
                                   out1 = linear1(out)
                                   out1 = relu1(out1)
                                   out2 = linear2(out1)
                                   out2 = rel2u(out2)

                                   A file contaning the replicated computation
                                   is automatically save in the output_folder.


                              -  replicate_with (type:str, optional,def:None):
                                  it can be used when we want to change
                                  some parameters during the replication
                                  process (e.g, when I wanna change the number
                                  of hidden neurons at each layer). The syntax
                                  to be used is the following:
                                  replicate_with=linear,n_neurons=512*2,256*1
                                  In this case we change the paremeter
                                  n_neurons of the linear function in the
                                  following way:
                                      layer 1: n_neurons=512
                                      layer 2: n_neurons=512
                                      layer 3: n_neurons=256


                            - add_connections (residual,skip,dense,
                                               "optional", "None"):
                                this flag can add additional connections
                                between computation blocks when replicating
                                the computations.
                                Residual connections are connections created
                                between two adjacent blocks. Skip connections
                                are created between each block and the final
                                one. Dense connections, insted are connections
                                created between the current block and all the
                                previous ones.

                            - connection_merge (sum,average,diff,concat,
                                                linear_comb, optional,
                                                def:sum):
                                when creating additional connections (see
                                add_connections) there are different ways
                                to combine the output of the blocks.
                                sum,average,diff operations can be used only
                                if the block have exactly the same
                                dimensionality. If this is not the case,
                                users can concatenated the outputs or
                                perform a linear (learnable) combination
                                among them.

                            - torch_no_grad (type:bool, optional, def:False):
                                If True, the computations will be performed
                                with the flag torch.no_grad() as required
                                in the test/validation modality.

                            - eval_mode (type:bool, optional, def:None):
                                If True, the computations will be performed
                                in eval modality. It could be useful for all
                                the computations that have different behaviour
                                in training/test modality (e.g., dropout or
                                batch normalization). The flag is extended to
                                all the child computations unless specified
                                differently.

                            - device (type:cpu,cuda, optional, def:cpu):
                                The computations will be performed on the
                                in specified device. The flag is extended to
                                all the child computations unless specified
                                differently.

                            - gpu_id (type:int(0,inf), optional, def:0):
                                When device=cuda, this flag can be used to
                                specify which cuda device to use. When multple
                                cuda devices are available this flag can be
                                used to perform model parallelization (e.g,
                                one can perform some computations on cuda:0 and
                                other on cuda:0).

                            - multi_gpu (type:bool, optional, def:None):
                                This fag can be used to perform data
                                parallelization over multiple gpus.
                                In practice, the batches of data created
                                from the csv file are split into n chunks.
                                For instance, if the batch_size=4 and we have
                                2 gpus, each gpu will see batches composed of
                                2 sentences. Each gpu processes the batches
                                in parallel. At the end, all the results are
                                combined in the reference gpu (by default
                                cuda:0). We suggest to use this function
                                only when the batch_size is large enough.
                                Otherwise, it's hard to see some speed up
                                over a single gpu.

                           - save_folder (type: path, optional, Default:None):
                               if the folder where to save the possible
                               outputs of the computation process.

                           - recovery (type: bool, optional, Default:False):
                               if True, the system restarts from the last
                               loop correctly executed.

                          - progress_bar: (type:bool, optional, def: False):
                              when true it adds a progress bar that can
                              be used to monitor the evolution of the
                              computations.

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

     Example:   # EX1: read audio file and add random noise
                from speechbrain.core import execute_computations

                exec_cfg={'class_name': 'speechbrain.py',
                          'cfg_file': 'cfg/minimal_examples/basic_processing/\
                           minimal_processing_read_write_example_noise.cfg',
                          'root_cfg': 'True'}

                # Initializing the execute computation class
                computations = execute_computations(exec_cfg)

                # Executing the computations specified in the config file
                computations([])

                # EX2: Computing MFCCs features
                exec_cfg={'class_name': 'speechbrain.py',
                          'cfg_file': 'cfg/minimal_examples/features/\
                          compute_mfccs_example.cfg',
                          'root_cfg': 'True'}

                # Initializing the execute computation class
                computations = execute_computations(exec_cfg)

                # Executing the computations specified in the config file
                computations([])


     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        class_name,
        cfg_file,
        csv_file=None,
        cfg_change=None,
        root_cfg=False,
        n_loops=1,
        stop_at=None,
        out_var=None,
        save_folder=None,
        recovery=False,
        accum_type=None,
        torch_no_grad=False,
        eval_mode=None,
        device=None,
        gpu_id=0,
        multi_gpu=None,
        batch_size=1,
        csv_read=None,
        sentence_sorting='original',
        num_workers=0,
        cache=False,
        cache_ram_percent=75,
        select_n_sentences=None,
        drop_last=False,
        replicate=1,
        replicate_with=None,
        add_connections=None,
        connection_merge='sum',
        progress_bar=False,
    ):

        # Here are summarized the expected options for this class
        options = [
            'class_name': {'type': 'file', 'value': class_name},
            'cfg_file': ('type': 'file', 'value': cfg_file},
            'csv_file': {'type': 'file', 'value': csv_file},
            'cfg_change': {'type': 'str', 'value': cfg_change},
            'root_cfg': {'type': 'bool', 'value': root_cfg},
            'n_loops': {'type': 'int(1,inf)', 'value': n_loops},
            'stop_at': {'type': 'str_list', 'value': stop_at},
            'out_var': {'type': 'str_list', 'value': out_var},
            'save_folder': {'type': 'str', 'value': save_folder},
            'recovery': {'type': 'bool', 'value': recovery},
            'accum_type': {
                'type': 'one_of_list(list,sum,average,last)',
                'value': accum_type,
            },
            'torch_no_grad': {'type': 'bool', 'value': torch_no_grad},
            'eval_mode': {'type': 'bool', 'value': eval_mode},
            'device': {'type': 'one_of(cuda,cpu)', 'value': device},
            'gpu_id': {'type': 'int_list(0,inf)', 'value': gpu_id},
            'multi_gpu': {'type': 'bool', 'value': multi_gpu},
            'batch_size': {'type': 'int(1,inf)', 'value': batch_size},
            'csv_read': {'type': 'str_list', 'value': csv_read},
            'sentence_sorting': {
                'type': 'one_of(ascending,descending,random,original)',
                'value': sentence_sorting,
            },
            'num_workers': {'type': 'int(0,inf)', 'value': num_workers},
            'cache': {'type': 'bool', 'value': cache},
            'cache_ram_percent': {
                'type': 'int(0,100)',
                'value': cache_ram_percent,
            },
            'select_n_sentences': {
                'type': 'int(1,inf)',
                'value': select_n_sentences,
            },
            'drop_last': {'type': 'bool', 'value': drop_last},
            'replicate': {'type': 'int(1,inf)', 'value': replicate},
            'replicate_with': {'type': 'str', 'value': replicate_with},
            'add_connections': {
                'type': 'one_of(residual,skip,dense)',
                'value': add_connections,
            },
            'connection_merge': {
                'type': 'one_of(sum,average,diff,concat,linear_comb)',
                'value': connection_merge,
            },
            'progress_bar': {'type': 'bool', 'value': progress_bar},
        }

        # Check, cast , and expand the options
        check_opts(options)

        # Setting global config
        if global_config is None:
            self.global_config = {}
        else:
            self.global_config = global_config

        # Declaring global function that gather all the functions
        # seen in all the config files
        if self.root_cfg:
            self.functions = {}
        else:

            if functions is None:
                self.functions = {}
            else:
                self.functions = functions

        # Reading the config file to execute
        self.config_proc = read_config(
            self.cfg_file,
            cfg_change=self.cfg_change,
            global_config=self.global_config,
            root_cfg=self.root_cfg,
        )

        # Creating logger and output directory (if needed)
        if "output_folder" in self.config_proc["global"]:
            self.output_folder = self.config_proc["global"]["output_folder"]
        else:
            self.output_folder = None

        if self.root_cfg:

            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

        if self.logger is None:
            # Logger initialization
            if self.output_folder is not None:
                log_file = self.output_folder + "/log.log"

                # Setup logger
                self.logger = setup_logger(
                    "logger",
                    log_file,
                    verbosity_stdout=self.config_proc["global"]["verbosity"],
                )
            else:
                log_file = None

        # Set verbosity
        if "verbose" in self.config_proc["global"]:
            self.verbose = self.config_proc["global"]["verbose"]
        else:
            self.verbose = True

        # Check if the specified device exists:
        if self.device == "cuda":

            if not torch.cuda.is_available():
                err_msg = (
                    'The function %s has specified "cuda" as device.'
                    "However, we cannot find any cuda-capable device."
                    % (self.funct_name,)
                )

                logger_write(err_msg, logfile=logger)

            # Check if the specified device exists
            for gpu in self.gpu_id:
                if gpu > torch.cuda.device_count() - 1:
                    err_msg = (
                        "The function %s has specified cuda:%s as device. "
                        "However, we cannot find it (we found %i gpus [0-%i])."
                        % (
                            self.funct_name,
                            gpu,
                            torch.cuda.device_count(),
                            torch.cuda.device_count() - 1,
                        )
                    )

                    logger_write(err_msg, logfile=logger)

            # Singe-gpu case
            if len(self.gpu_id) == 1:
                self.gpu_id = self.gpu_id[0]

        # Creating dataloader for the specified csv file
        if self.csv_file is not None:

            dataloader_obj = create_dataloader(
                config, global_config=self.global_config, logger=self.logger
            )
            self.dataloader = dataloader_obj.dataloader
            self.ite_num = dataloader_obj.data_len / self.batch_size

            # Updating csv read
            self.csv_read = dataloader_obj.csv_read

            # Adding label dict in global. This way will be visible by all
            # the config files called.
            self.global_config["label_dict"] = dataloader_obj.label_dict

        # Selecting recovery path
        if self.n_loops >= 1:
            self.start_index = self.recover_iteration()
        else:
            self.start_index = 0

        # Initializing a variable that will gather all the neural parameters
        self.params = torch.nn.ModuleList([])

        # List of all the possible functions that we can call in the
        # computation section
        all_funct = list(self.config_proc["functions"].keys()) + list(
            self.functions.keys()
        )

        all_funct = list(set(all_funct))

        # Getting command string from the computation section
        cmd_str = self.config_proc["computations"]

        # Replace all the functions met in the computation section with
        # self.run_function
        cmd_str = process_cmd_string(cmd_str, all_funct)

        # Scanning the computation to decide at which computation
        # the needed variables in self.stop_at can be returned.
        stop_computations = self.detect_stop_computations(
            cmd_str, self.stop_at
        )

        # Check at which computation the output variables can be returned
        last_out_var_computation = self.detect_stop_computations(
            cmd_str, self.out_var
        )

        # Make sure accum_type has the same length of out_var when defined
        if self.out_var is not None:

            if self.accum_type is not None:

                if len(self.accum_type) != len(self.out_var):

                    err_msg = (
                        "the field accum_type must be a list with the "
                        "same  length  of the one specified in out_var."
                        "Got %i vs %i"
                        % (len(self.accum_type), len(self.out_var))
                    )

                    logger_write(err_msg, logfile=logger)
            else:

                self.accum_type = ["list" for i in range(len(self.out_var))]

        else:
            self.out_var = []

        # Stopping the computation when encoutering the variables in
        # self.stop_at. If the user requires a variable that comes after
        # that, by default we perform the computation until all the out_var
        # are founded (this is implemented by the following max).

        self.stop_computations = max(
            stop_computations, last_out_var_computation
        )

        # if both stop_at and out_var are None, we will do all the computations
        if self.stop_computations == -1:
            self.stop_computations = len(cmd_str.split("\n")) - 1

        # Selecting only the needed part of the computations
        cmd_str = "\n".join(cmd_str.split("\n")[0: self.stop_computations])

        # Manage replicate functionality (useful to replicate some blocks of
        # computations like when adding layers to a neural network)

        if self.replicate > 1:
            cmd_str = self.manage_replicate(cmd_str)

        # Parsing the computations
        cmd_parsed = ast.parse(cmd_str)

        # Compilation of the computations to execute
        self.cmd = compile(cmd_parsed, filename="cmd", mode="exec")

        # Gathering the variables to pass to the python shell for computations
        self.vars = {**globals(), **locals()}

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

        # Create saving folder if needed:
        if self.save_folder is not None:
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

    def forward(self, inp):

        # Initializing the output_var list
        self.out_var_lst = []

        # Managing accumulation of the output
        if len(self.out_var) > 0:

            for i, var in enumerate(self.out_var):

                if (
                    self.accum_type[i] == "list"
                    or self.accum_type[i] == "last"
                ):

                    self.out_var_lst.append([])

                if (
                    self.accum_type[i] == "sum"
                    or self.accum_type[i] == "average"
                ):

                    self.out_var_lst.append(0)

        # Processing the data for n_loops
        for i in range(self.start_index, self.n_loops):

            # Managing data loops where we loop also over the data in the csv
            # file
            if self.csv_file is not None:

                # Initialization of the progress bar
                if self.progress_bar:
                    pbar = tqdm(total=int(self.ite_num))

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
                    batch = batch + list(inp)

                    # Processing the batch as specified in self.cfg_file
                    self.run_computations(batch)

                    # Progress bar
                    if self.progress_bar:
                        pbar.update(1)

                # Closing the progress bar
                if self.progress_bar:
                    pbar.close()

                # Appending the last output to out_var_lst if needed
                if len(self.out_var) > 0:
                    for out_id, out in enumerate(self.output):

                        if self.accum_type[out_id] == "last":
                            self.out_var_lst[out_id] = out

                        if self.accum_type[out_id] == "average":
                            N_elements = (i + 1) * (batch_id + 1)

                            self.out_var_lst[out_id] = (
                                self.out_var_lst[out_id] / N_elements
                            )

            # Managing standard loop where we only replicate the computations
            # n_loops time (without looping also over the csv data)
            else:
                # Running the computations reported in self.cfg_file
                inp_comp = list(inp)
                inp_comp.append(i)
                self.run_computations(inp_comp)

            # Store iteration (if needed)
            self.store_iteration(i)

        # Saving the output variables when required
        if self.save_folder is not None:

            for index, out in enumerate(self.out_var_lst):

                out_file = (
                    self.output_folder
                    + "/"
                    + self.funct_name
                    + "/"
                    + self.ot_var[index]
                    + ".pkl"
                )

                save_pkl(out, out_file)

        # Return a single element when list has len=1
        if isinstance(self.out_var_lst, list):
            if len(self.out_var_lst) == 1:
                self.out_var_lst = self.out_var_lst[0]

        return self.out_var_lst

    def run_computations(self, inp):
        """
         ---------------------------------------------------------------------
         core.execute_computations.run_computations (author: Mirco Ravanelli)

         Description: This function executes the computations specified
                      in the cfg_file file. It also stores the
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

         Example:   from speechbrain.core import execute_computations

                    exec_cfg={'class_name': 'speechbrain.py',
                              'cfg_file': 'cfg/minimal_examples/\
                              basic_processing/\
                              minimal_processing_read_write_example_noise.cfg',
                              'root_cfg': 'True'}

                    # Initializing the execute computation class
                    computations = execute_computations(exec_cfg)

                    # Execute computations
                    computations.run_computations([])

         ---------------------------------------------------------------------
         """

        # Running the computations reported in the cfg_file file
        if self.cfg_file is not None:

            # Adding torch_no_grad flag when needed
            if self.torch_no_grad:
                with torch.no_grad():
                    self.output = self.exec_computations(inp)

            else:
                self.output = self.exec_computations(inp)

        # Converting output to list
        if isinstance(self.output, torch.Tensor):
            self.output = [self.output]

        # Storing the output in a list if required
        if len(self.out_var) > 0 and (
            self.n_loops > 1 or self.csv_file is not None
        ):

            # Detatching torch.Tensors before accumulation
            self.output = self.detach_tensor_in_list(self.output)

            # Accumulating outputs as specified in accum_type
            for i, out in enumerate(self.output):

                # Appending out_var in a list
                if self.accum_type[i] == "list":
                    self.out_var_lst[i].append(out)

                # Summing up the returned out_vars
                if (
                    self.accum_type[i] == "sum"
                    or self.accum_type[i] == "average"
                ):

                    self.out_var_lst[i] = self.out_var_lst[i] + out
        else:
            self.out_var_lst = self.output

    @staticmethod
    def detach_tensor_in_list(lst):
        """
         ---------------------------------------------------------------------
         core.execute_computations.detach_tensor_in_list (author: M. Ravanelli)

         Description: This scans a list and detaches the tensors in it

         Input:        - lst (type:list, mandatory):
                         it is a list whose tensors need to be detached.


         Output:      - out_lst (type: list):
                         it is the output list with the tensor detached

         Example:   import torch
                    from speechbrain.core import execute_computations

                    exec_cfg={'class_name': 'speechbrain.py',
                              'cfg_file': 'cfg/minimal_examples/\
                              basic_processing/\
                              minimal_processing_read_write_example_noise.cfg',
                              'root_cfg': 'True'}

                    # Initializing the execute computation class
                    computations = execute_computations(exec_cfg)

                    # Example of a list
                    data_list=[torch.Tensor([1,2,3,4]),'a','b']
                    # list conversion
                    data_list_out=computations.detach_tensor_in_list(data_list)

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
         core.execute_computations.prepare_data_list (author: Mirco Ravanelli)

         Description: This function converts the data read by the
                      data_loader into a list formatted in this way:
                      [snt_ID, data, data_len, data2, data_len2,...]

         Input:       - data_list (type, list, mandatory)


         Output:      - data_list (type: list)


         Example:   import torch
                    from speechbrain.core import execute_computations

                    exec_cfg={'class_name': 'speechbrain.py',
                              'cfg_file': 'cfg/minimal_examples/\
                              basic_processing/\
                              minimal_processing_read_write_example_noise.cfg',
                              'root_cfg': 'True'}

                    # Initializing the execute computation class
                    computations = execute_computations(exec_cfg)

                    # data_list (example of what returned by the dataloader)
                    data_list=[[['example1'],\
                                torch.tensor([1,2,3,4]),\
                                torch.tensor(1.0)], \
                                [['example1'],\
                                 [['spk01']],\
                                 torch.tensor(1.0)]]
                    # list conversion
                    data_list_out=computations.prepare_data_list(data_list)
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

    def recover_iteration(self,):
        """
         ---------------------------------------------------------------------
         core.execute_computations.recover_iteration (author: Mirco Ravanelli)

         Description: This function checks the recovery dict stored in
                      output_folder/recovery.pkl and outputs the last
                      iteration that has benn correctly finished.

         Input:       None


         Output:      - start_index (type: int):
                           it is the index of the last iteration correctly
                           finished.

         Example:   import torch
                    from speechbrain.core import execute_computations

                    exec_cfg={'class_name': 'speechbrain.py',
                              'cfg_file': 'cfg/minimal_examples/\
                                  neural_networks/spk_id/spk_id_example.cfg',
                              'root_cfg': 'True'}

                    # Initializing the execute computation class
                    computations = execute_computations(exec_cfg)

                    # running the computations
                    computations([])

                    # printing the last iteration correctly done
                    print(computations.functions["training_nn"]\
                        .recover_iteration())

         ---------------------------------------------------------------------
         """

        # Initalizing the starting index
        self.start_index = 0

        # Recover last index only if recovery is set to True
        if self.recovery:

            # Setting the recovery path
            self.recovery_path = (
                self.global_config["output_folder"] + "/recovery.pkl"
            )

            if os.path.exists(self.recovery_path):

                # Loading the recovery dictionary
                self.recovery_dict = load_pkl(self.recovery_path)

                if self.funct_name in self.recovery_dict:

                    # Looking for the 'loop_id' keyword
                    if "loop_id" in self.recovery_dict[self.funct_name]:

                        # Getting the id of the last loop finished
                        self.start_index = (
                            self.recovery_dict[self.funct_name]["loop_id"] + 1
                        )

        return self.start_index

    def store_iteration(self, index):
        """
        ---------------------------------------------------------------------
        core.execute_computations.store_iteration (author: Mirco Ravanelli)

        Description: This function keeps track of the iterations correctly
                     finished by writing the loop_id into the recovery
                     dictionary in output_folder/recovery.pkl.

        Input:       - index (type:int):
                        it is the index of the current iteration.


        Output:      - None:
                        the function directly write into the recovery dict.

        Example:   import torch
                   from speechbrain.data_io.data_io import load_pkl
                   from speechbrain.core import execute_computations

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': 'cfg/minimal_examples/\
                             neural_networks/spk_id/spk_id_example.cfg',
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   # running the computations
                   computations([])

                   print(computations.functions["training_nn"]\
                       .recover_iteration())

                   # adding an iteration
                   computations.functions["training_nn"].store_iteration(4)

                   print(computations.functions["training_nn"]\
                       .recover_iteration())

        ---------------------------------------------------------------------
        """

        # Writing the iteration index only if recovery is True
        if self.recovery:

            # Path of the recovery dictionary
            self.recovery_path = (
                self.global_config["output_folder"] + "/recovery.pkl"
            )

            if os.path.exists(self.recovery_path):

                # Loading the recovery dictionary
                self.recovery_dict = load_pkl(self.recovery_path)

                if self.funct_name in self.recovery_dict:
                    self.recovery_dict[self.funct_name]["loop_id"] = {}
                else:
                    self.recovery_dict[self.funct_name] = {}
                    self.recovery_dict[self.funct_name]["loop_id"] = {}

                # Saving iteration id
                self.recovery_dict[self.funct_name]["loop_id"] = index

            else:

                # Creating the recovery dictionary if it does not exist
                self.recovery_dict = {}

                self.recovery_dict[self.funct_name] = {}

                self.recovery_dict[self.funct_name]["loop_id"] = index

            # Saving the dictionary
            save_pkl(self.recovery_dict, self.recovery_path)

        return

    def exec_computations(self, inp_lst):
        """
        ---------------------------------------------------------------------
        core.execute_computations.exec_computations (author: Mirco Ravanelli)

        Description: This function execute (with the exec command) the
                     compiled computations in self.cmd.

        Input:       - index (type:inp_last):
                        it is a list containing the inputs


        Output:      - out_var_lst (type:list):
                        the function returs all the specified output variables
                        in a list.

        Example:   import torch
                   from speechbrain.data_io.data_io import load_pkl
                   from speechbrain.core import execute_computations

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': 'cfg/minimal_examples/\
                             neural_networks/spk_id/spk_id_example.cfg',
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   # running the computations
                   computations.exec_computations([])

        ---------------------------------------------------------------------
        """
        # Reading the input list
        self.input_var = inp_lst

        # Local variables
        local_variables = {}

        # Execution of the computations
        exec(self.cmd, self.vars, local_variables)

        # Gathering all the output variables in a list
        out_var_lst = []

        if len(self.out_var) > 0:

            # Return in output a list as specified in out_var
            out_var_lst = []

            for var in self.out_var:
                out_var_lst.append(local_variables[var])

        else:
            # Return the last variable if out_var is not specified
            if len(local_variables) > 0:
                last_elem = list(local_variables.keys())[-1]
                out_var_lst = [local_variables[last_elem]]
            else:
                out_var_lst = [None]

        # Returning a single element if len(out_var_lst)=1
        if len(out_var_lst) == 1:
            out_var_lst = out_var_lst[0]

        return out_var_lst

    def run_function(self, *argv):
        """
         ---------------------------------------------------------------------
         core.execute_computations.run_function
         (author: Mirco Ravanelli)

         Description: This function executes the given computation. This
                      funcion takes in input a list with a variable number
                      of arguments. By default the first argument is the
                      name of the function to run, while the others are the
                      input that must be provided to the the function. The
                      output is a list containing the returned variables.
                      Run_function also supports a special functions called
                      get_inp_var() that simply returns the input variables
                      given when calling the execute_computation class.
                      The function is stored into the self.function dictionary
                      and it is initialized only the first time it is called.

         Input:        - self (type, execute_computaion class, mandatory)

                       - argv (type, list, mandatory):
                           it is a list that contains the name of the
                           function to run followed by all its input args

         Output:         - result (type, list):
                            it is a list containing all the values returned
                            by the executed function.


         Example:    from speechbrain.core import execute_computations

                     cfg='cfg/minimal_examples/data_reading/\
                     loop_example.cfg'

                     # Definition of the exec_config dictionary
                     exec_config={'class_name':'core.execute_computations', \
                                  'cfg_file': cfg, \
                                  'root_cfg':'True'}

                    # Initialization of the class
                    computations=execute_computations(exec_config)

                    # Running the first function:
                    computations.run_function('loop')
        """

        # Getting the function name (first argument by default)
        funct_name = argv[0]

        # Initialization of the class (if called for the first time)
        if funct_name not in self.functions:

            if funct_name in self.config_proc["functions"]:
                library = self.config_proc["functions"][funct_name][
                    "class_name"
                ]
                del self.config_proc["functions"][funct_name]["class_name"]

                # Importing Library
                try:
                    lib = import_class(library)
                except Exception:
                    err_msg = "Cannot import class %s" % (library)
                    logger_write(err_msg, logfile=self.logger, level="debug")
                    raise

                # Function Initialization
                try:
                    self.functions[funct_name] = lib(
                        **self.config_proc["functions"][funct_name],
                        global_config=self.global_config,
                        logger=self.logger,
                    )

                    # Keeping track of parent name
                    self.functions[funct_name].parent_name = self.funct_name

                    # Switch function to the right device
                    self.to_device(funct_name)

                    # Managing multiple gpu
                    self.manage_multi_gpu_init(funct_name, lib, argv)

                except Exception:
                    err_msg = "Cannot initialize function %s" % (funct_name)
                    logger_write(err_msg, logfile=self.logger, level="debug")
                    raise

                # If function is a neural network, add its parameters in
                # self.params
                if hasattr(self.functions[funct_name], "parameters"):
                    self.params.append(self.functions[funct_name])

        # Setting eval modality (when needed)
        self.set_eval_mode(funct_name)

        # Reading the inputs
        inp_lst = list(argv[1:])

        # Input to device
        inp_lst = self.inp_to_device(funct_name, inp_lst)

        # Updating parent name
        self.functions[funct_name].parent_name = self.funct_name

        if (
            hasattr(self.functions[funct_name], "multi_gpu")
            and self.functions[funct_name].multi_gpu
        ):

            # Running multi-gpu computations
            result = self.multi_gpu_computation(funct_name, inp_lst)

        else:
            # Running computations
            result = self.functions[funct_name](inp_lst)

        return result

    def to_device(self, funct_name):
        """
        ---------------------------------------------------------------------
        core.execute_computations.to_device (author: Mirco Ravanelli)

        Description: This function set the specified designed to the
                     function funct_name. When the device is not set,
                     we inherit the device from the parent function.

        Input:       - funct_name (type:str):
                        it is the name of the function that must be
                        executed on the selected device.


        Output:      - None

        Example:   import torch
                   from speechbrain.core import execute_computations

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': 'cfg/minimal_examples/\
                             neural_networks/spk_id/spk_id_example.cfg',
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   # running computations of the cpu
                   computations([])

                   computations.functions['training_nn'].device = 'cuda'
                   computations.functions['training_nn'].gpu_id = 0

                   computations.to_device('training_nn')

                   # running the computations on the gpu
                   computations.exec_computations([])

        ---------------------------------------------------------------------
        """

        # Managing devices
        funct_device = "cpu"
        inherit_parent_device = False

        # Check if the attribute device is specified
        if hasattr(self.functions[funct_name], "device"):

            # If device is specified use it
            if self.functions[funct_name].device is not None:

                if "cuda" in self.functions[funct_name].device:
                    funct_device = "cuda:" + str(
                        self.functions[funct_name].gpu_id
                    )

            # If device is specified, use the parent device
            else:
                inherit_parent_device = True

        else:
            inherit_parent_device = True

        # Inheriting the parent device
        if inherit_parent_device:
            parent_name = self.functions[funct_name].parent_name
            if parent_name is not None:
                if hasattr(self.functions[parent_name], "device"):
                    funct_device = self.functions[parent_name].device

        # Putting the function on the selected device
        if hasattr(self.functions[funct_name], "to"):
            self.functions[funct_name].device = funct_device
            self.functions[funct_name].to(funct_device)

    def inp_to_device(self, funct_name, inp_lst):
        """
        ---------------------------------------------------------------------
        core.execute_computations.inp_to_device (author: Mirco Ravanelli)

        Description: this support function put the inputs of the same
                     device of the function funct_name that is going to
                     be executed.

        Input:       - funct_name (type:str):
                        it is the name of the function that must be
                        executed.

                     - inp_lst (type:list):
                         it is the list contaning the inputs



        Output:      - inp_lst_device (type:list):
                        the function returs all the inputs on the selected
                        device.

        Example:   import torch
                   from speechbrain.core import execute_computations

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': 'cfg/minimal_examples/\
                                 neural_networks/spk_id/spk_id_example.cfg',
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   # running computations of the cpu
                   computations([])

                   computations.functions['training_nn'].device = 'cuda'

                   # random input (on cpu)
                   inp_tensor = torch.rand(2,2,2)
                   print(inp_tensor)

                   inp_dev=computations.inp_to_device('training_nn',
                   [inp_tensor])
                   print(inp_dev)

        ---------------------------------------------------------------------
        """

        # Loading the input to device (if needed)
        if hasattr(self.functions[funct_name], "device"):

            inp_lst_device = []

            # Looping over all the inputs
            for inp in inp_lst:
                if isinstance(inp, torch.Tensor):
                    inp = inp.to(self.functions[funct_name].device)

                inp_lst_device.append(inp)
                inp_lst = inp_lst_device

            return inp_lst_device
        else:
            return inp_lst

    def set_eval_mode(self, funct_name):
        """
        ---------------------------------------------------------------------
        core.execute_computations.set_eval_mode (author: Mirco Ravanelli)

        Description: this method switches the current to eval or training
                     modality. When the modality is not specified, the
                     modality of the parent function is selected.

        Input:       - funct_name (type:str):
                        it is the name of the function that must be
                        executed.


        Output:      None


        Example:   import torch
                   from speechbrain.core import execute_computations

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': 'cfg/minimal_examples/\
                                 neural_networks/spk_id/spk_id_example.cfg',
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   # running computations with training modality
                   computations([])

                   # changing to eval modality
                   computations.functions['training_nn'].eval_mode = True

                   computations.set_eval_mode('training_nn')
                   computations([])


        ---------------------------------------------------------------------
        """

        # Variable initialization
        funct_eval_mode = False
        inherit_parent_mode = False
        parent_name = self.funct_name

        # Check if the function has the attribute eval_mode
        if hasattr(self.functions[funct_name], "eval_mode"):

            # Use the modality specified in the field eval_mode
            if self.functions[funct_name].eval_mode is not None:
                funct_eval_mode = self.functions[funct_name].eval_mode

            # If the modality is not specified, inherit it from the parent
            else:
                inherit_parent_mode = True

        else:
            inherit_parent_mode = True

        # Inheriting modality from the parent
        if inherit_parent_mode:

            if parent_name is not None:
                if hasattr(self.functions[parent_name], "funct_eval_mode"):
                    funct_eval_mode = self.functions[
                        parent_name
                    ].funct_eval_mode

        # Setting the modality (only for neural models)
        if hasattr(self.functions[funct_name], "to"):
            self.functions[funct_name].funct_eval_mode = funct_eval_mode
            if funct_eval_mode:
                self.functions[funct_name].eval()
            else:
                self.functions[funct_name].train()

        # Managing multiple gpus
        if hasattr(self.functions[funct_name], "multi_gpu_model_names"):
            if len(self.functions[funct_name].multi_gpu_model_names) > 0:
                for funct_gpu_name in self.functions[
                    funct_name
                ].multi_gpu_model_names:
                    self.functions[
                        funct_gpu_name
                    ].funct_eval_mode = self.functions[
                        funct_name
                    ].funct_eval_mode

    def manage_multi_gpu_init(self, funct_name, lib, argv):
        """
        ---------------------------------------------------------------------
        core.execute_computations.manage_multi_gpu_init (author: M. Ravanelli)

        Description: this function initializes the computations on multiple
                     gpus.  When the multi-gpu flag is True, the computations
                     are replicated on the different selected devices. For
                     instance, if there is a function 'linear' in the
                     computation section, we will also create linear_cuda:1,
                     linear_cuda:2, linear_cuda:3 when we have 4 gpus.
                     The cuda:0 device is by default the reference device
                     that will collected and combine the results of all
                     the other gpus.

        Input:       - funct_name (type:str):
                        it is the name of the function that must be
                        executed.

                     - lib (type:str):
                         it is the library used to initialize again
                         the funct_name function.

                     - argv (type:list):
                         it is a list containing the first input values.


        Output:      None
                        the method directly adds into self.functions the
                        new functions initialized on the gpus.

        ---------------------------------------------------------------------
        """

        # Check the multi_gpu attribute
        if hasattr(self.functions[funct_name], "multi_gpu"):

            if self.functions[funct_name].multi_gpu:

                # When the multi_gpu attribute is True,  intialize the
                # functions
                self.functions[funct_name].multi_gpu_models = []
                self.functions[funct_name].multi_gpu_model_names = []

                # Looping over all the devices selected
                for device_id in self.functions[funct_name].gpu_id[1:]:

                    # Initialize the function on the current gpu
                    self.initialize_multi_gpu(
                        funct_name, "cuda:" + str(device_id), lib, argv[1:]
                    )

                    # Updating multi_gpu_models flag with the current model
                    self.functions[funct_name].multi_gpu_models.append(
                        self.functions[funct_name + "_cuda:" + str(device_id)]
                    )
                    self.functions[funct_name].multi_gpu_model_names.append(
                        funct_name + "_cuda:" + str(device_id)
                    )

            # Getting the parent function
            parent_name = self.functions[funct_name].parent_name

            # Inheriting the parent device when needed
            if parent_name is not None:
                if hasattr(self.functions[parent_name], "multi_gpu"):
                    if self.functions[parent_name].multi_gpu:
                        if self.functions[funct_name].multi_gpu is None:
                            for device_id in self.functions[
                                parent_name
                            ].gpu_id[1:]:
                                if "_cuda:" + str(device_id) in parent_name:
                                    self.initialize_multi_gpu(
                                        funct_name,
                                        "cuda:" + str(device_id),
                                        lib,
                                        argv[1:],
                                    )

    def initialize_multi_gpu(self, funct_name, device_id, lib, inp_lst):
        """
        ---------------------------------------------------------------------
        core.execute_computations.initialize_multi_gpu (author: M. Ravanelli)

        Description: this function initializes the function funct_name on
                     another gpu device with the name funct_name_cuda:dev_id.

        Input:       - funct_name (type:str):
                        it is the name of the function that must be
                        executed.

                     - device_id (type: int(0,inf)):
                         it is the id of the device where we want to
                         initialize the function funct_name.

                    - lib (type: str):
                        it is the library that we have to use to initialize
                        the function.

                     - inp_lst (type:list):
                         it is a list containing all the input elements
                         to process.



        Output:      None
                        this method initializes the function in the
                        self.function dictionaries with the following
                        name: 'funct_name_cuda:device_id'. Moreover,
                        the config file that each gpu runs is
                        saved in output_folder.

        ---------------------------------------------------------------------
        """

        # Getting the function name
        if "_cuda:" not in funct_name:
            funct_name_gpu = funct_name + "_" + str(device_id)
        else:
            funct_name_gpu = funct_name

        # Change function names in the cfg file:
        orig_file = self.functions[funct_name].cfg_file
        dest_file = self.output_folder + "/" + funct_name_gpu + ".cfg"

        # Creating the cfg_file for the current device in the output_folder
        self.change_funct_name(orig_file, dest_file, "_" + str(device_id))

        # copy the configuration file
        config_proc_gpu = self.config_proc["functions"][funct_name].copy()
        config_proc_gpu["cfg_file"] = dest_file

        # Function initialization
        self.functions[funct_name_gpu] = lib(
            config_proc_gpu,
            funct_name=funct_name_gpu,
            global_config=self.global_config,
            functions=self.functions,
            logger=self.logger,
            first_input=inp_lst,
        )

        # setting function to device
        self.functions[funct_name_gpu].device = device_id

        # Keeping track of parent name
        self.functions[funct_name_gpu].parent_name = self.funct_name

        # setting function to device
        self.functions[funct_name_gpu].to(device_id)

    def multi_gpu_computation(self, funct_name, inp_lst):
        """
        ---------------------------------------------------------------------
        core.execute_computations.multi_gpu_computation (author: M. Ravanelli)

        Description: this function performs the computations on multiple
                     gpus. It splits the batches of data over the specified
                     gpus and process all the batches in parallel. The results
                     are gathered into the reference gpu cuda:0.

        Input:       - funct_name (type:str):
                        it is the name of the function that must be
                        executed.

                     - inp_lst (type:list):
                         it is a list containing all the input elements
                         to process.



        Output:      result
                        the functions return the results of the computations
                        already gathered into the reference gpu cuda:0

        ---------------------------------------------------------------------
        """

        funct_lst = []
        funct_lst.append(self.functions[funct_name])

        # Copy all the parameters of the reference device on all the other
        # gpus.
        for gpu in self.functions[funct_name].gpu_id[1:]:

            # getting funct_name on the current gpu
            funct_name_gpu = funct_name + "_cuda:" + str(gpu)

            # copy parameters
            if hasattr(self.functions[funct_name], "parameters"):
                if len(list(self.functions[funct_name].parameters())) > 0:
                    # copy parameters:
                    self.functions[funct_name_gpu].load_state_dict(
                        self.functions[funct_name].state_dict()
                    )
                    self.functions[funct_name_gpu].zero_grad()

            # Adding the current function into the current funct list
            funct_lst.append(self.functions[funct_name_gpu])

        # Split the input over the batch axis
        inputs = nn.parallel.scatter(
            inp_lst, [0] * len(self.functions[funct_name].gpu_id)
        )

        # Formatting the inputs are required by nn.parallel.parallel_apply
        for inp_id, inp in enumerate(inputs):
            inputs[inp_id] = (inputs[inp_id],)

        # Processing all the data in parallel
        outputs = nn.parallel.parallel_apply(funct_lst, inputs)

        # Gathering all the results on the reference gpu
        result = nn.parallel.gather(
            outputs, self.functions[funct_name].gpu_id[0]
        )

        return result

    def change_funct_name(self, orig_file, dest_file, new_name):
        """
        ---------------------------------------------------------------------
        core.execute_computations.change_funct_name (author: Mirco Ravanelli)

        Description: this support functions takes in input a configuration
                     file and appends to the name of each function the
                     pattern new_new. It is used in the multi-gpu case
                     to replicate the computations on multiple devices.

        Input:       - origin_file (type:path):
                        it is the configuration file to change.

                     - dest_file (type:path):
                         it is the new configuration file where the
                         modications are saved.

                    - new_name (type:str):
                        it is the patter to append to each function_name

        Output:      None
                        the output is directly written into the dest_file.

        Example:   import torch
                   from speechbrain.core import execute_computations

                   cfg_file = 'cfg/minimal_examples/neural_networks/spk_id/\
                       spk_id_example.cfg'

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': cfg_file,
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   dest_file = 'new_cfg_file.cfg'

                   computations.change_funct_name(cfg_file,dest_file,'_cuda:1')

                   # Now, you can open dest_file and check the name change.

        ---------------------------------------------------------------------
        """

        # Reading the config file
        with open(orig_file) as f:
            lines = f.readlines()

        # Getting all the function names in the config file
        funct_name_lst = []

        # Processing all the lines
        for line in lines:
            if "[/" in line and "]" in line:
                if "global" not in line:
                    if "functions" not in line:
                        if "computations" not in line:
                            funct_name_lst.append(
                                line.replace("[/", "")
                                .replace("]", "")
                                .replace("\t", "")
                                .replace(" ", "")
                                .replace("\n", "")
                            )

        # Replacing the funct_name with funct_name+new_name
        lines_new_cfg = []
        for line in lines:

            if "[" in line and "]" in line or "(" in line and ")" in line:
                for name in funct_name_lst:
                    line = line.replace(name, name + new_name)

            lines_new_cfg.append(line)

        # Saving the modified configuration file
        with open(dest_file, "w") as f:
            for item in lines_new_cfg:
                f.write("%s" % item)

    def get_input_var(self):
        """
        ---------------------------------------------------------------------
        core.execute_computations.get_input_var (author: Mirco Ravanelli)

        Description: this can be use to output the current input variables.
                     It is useful when a csv_file is specified and the
                     execute function will loop over all the data batches.
                     In the processing cfg_file (see for instance
                     cfg/minimal_examples/basic_processing\
                         /save_signals_with_noise.cfg
                     that is called from
                     cfg/minimal_examples/basic_processing/\
                         minimal_processing_read_write_example_noise.cfg)
                     the computation section contains the following line:
                     id,wav,wav_len,*_=get_input_var()
                     This means that we import in the current computation
                     section the id of the sentence followed by all the
                     batches of data specified in csv_read.


        Input:       None
        Output:      input_var
                        - the function returns the input variables.

        ---------------------------------------------------------------------
        """
        return self.input_var

    def detect_stop_computations(self, cmd_str, stop_var):
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
                      - stop_computations (type: list, mandatory):
                          it is a list of variables where to stop the
                          computations.


         Output:      - stop_id (type: str):
                            it is the line where we have to stop the
                            computations.

         Example:    from speechbrain.core import execute_computations

                   cfg_file = 'cfg/minimal_examples/neural_networks/spk_id/\
                       spk_id_example.cfg'

                   exec_cfg={'class_name': 'speechbrain.py',
                             'cfg_file': cfg_file,
                             'root_cfg': 'True'}

                   # Initializing the execute computation class
                   computations = execute_computations(exec_cfg)

                   # Initializing some "fake computations"
                   cmd_str = 'a=funct1(wav)\nb=funct2(a)\nc=funct3(a,b)'

                   print(cmd_str)

                   # Stopping computations when b is met:
                   stop_index = computations.detect_stop_computations(cmd_str,\
                   ['b'])

                   print(stop_index)
                   print(cmd_str.split('\n')[0:stop_index])

         ---------------------------------------------------------------------
         """

        # If stop_var is None, we will return -1 as computation_id
        stop_computations = -1

        if stop_var is not None and len(stop_var) > 0:

            stop_computations = []

            # Parsing the computations
            cmd_parsed = ast.parse(cmd_str)

            # Exploring the tree and look for the stop variables
            for node in ast.walk(cmd_parsed):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and node.id in stop_var
                ):

                    # Saving the line number where stop var is met
                    stop_computations.append(node.lineno)

                elif isinstance(node, ast.Str) and node.s in stop_var:

                    # Saving the line number where stop var is met
                    stop_computations.append(node.lineno)

            # Check if the all the out variables are encountered
            if len(stop_computations) == 0:
                err_msg = (
                    "the output variables (stop_var=%s) defined in "
                    "execute_computations are not defined in the section  "
                    "[computation] of the config file!" % (stop_var)
                )

                logger_write(err_msg)

            # Returning the last computation where the variable is assigned
            stop_computations = max(stop_computations)

        return stop_computations

    def manage_replicate(self, cmd_str):
        """
         ---------------------------------------------------------------------
         core.execute_computations.detect_stop_computations
         (author: Mirco Ravanelli)

         Description: This function replicates the computations reported
                      in a configuration file. It can be used to create
                      bigger neural networks by replicating multiple time
                      the same types of computations (e.g, adding layers
                      to a neural architecture)

         Input:       - cmd_str (type: string, mandatory)
                         it is a string contaning the basic computation
                         to replicate.


         Output:      - cmd_str (type: str)
                         it is the new string the replicated computations

         Example:    from speechbrain.core import execute_computations

                     cfg_file='cfg/minimal_examples/neural_networks/spk_id/\
                         spk_id_example.cfg'

                     # Definition of the exec_config dictionary
                     exec_config={'class_name':'core.execute_computations', \
                                  'cfg_file': cfg_file}

                     # Initialization of the class
                     computations=execute_computations(exec_config)

                     # Create a "fake" command to replicate
                     cmd_str='out=linear(wav)\nout=activ(out)'
                     print(cmd_str)

                     # set the number of replicas
                     computations.replicate=5
                     computations.out_var=['out']

                     cmd_str_repl=computations.manage_replicate(cmd_str)

                     # added connections
                     print(cmd_str_repl)

         ---------------------------------------------------------------------
         """

        # Creating replace dictionary
        self.create_replace_dict()

        # Creating a new config dictionary
        self.config_proc_replicate = {}
        self.config_proc_replicate["global"] = self.config_proc["global"]

        # Replicate functions
        self.replicate_functions()

        # Replicate computations
        cmd_to_replicate, current_inp = self.detect_cmd_to_replicate(cmd_str)
        self.replicate_computations(cmd_to_replicate, current_inp)

        # Saving the output
        conf_text = conf_to_text(self.config_proc_replicate)

        config_fn = self.cfg_file.split("/")[-1].replace(".", "_replicate.")

        if "output_folder" in self.global_config:
            write_config(
                conf_text,
                self.global_config["output_folder"] + "/" + config_fn,
                modality="w",
            )

        self.config_proc = self.config_proc_replicate

        # getting the replicated string of commands
        cmd_str = self.config_proc_replicate["computations"]

        return cmd_str

    def replicate_computations(self, cmd_to_replicate, current_inp):
        """
         ---------------------------------------------------------------------
         core.execute_computations.replicate_computations
         (author: Mirco Ravanelli)

         Description: This support function takes in input the command string
                      to replicate and stores the computations in
                      self.config_proc_replicate['computations'].

         Input:       - cmd_to_replicate (type: string, mandatory)
                         it is a string contaning the computations to
                         replicate.

                     - current_inp (type: string, mandatory)
                         it is a string containing the variable in input
                         to the computation block.


         Output:      - None
                         the output is saved in self.config_proc_replicate\
                             ['computations']
         ---------------------------------------------------------------------
        """

        out_lst = []

        lin_comb_cnt = 0

        # Replicating the computations
        for block_id in range(self.replicate):

            cmd_append = cmd_to_replicate

            # replacing function name
            for funct_name in self.config_proc["functions"]:
                cmd_append = cmd_append.replace(
                    '"' + funct_name + '"',
                    '"' + funct_name + "_" + str(block_id) + '"',
                )

            cmd_append = cmd_append.replace("$current_inp", current_inp)

            cmd_append = cmd_append.replace(
                "$current_out", self.out_var[0] + "_" + str(block_id)
            )

            out_lst.append(self.out_var[0] + "_" + str(block_id))

            # Adding residual connection:
            if self.add_connections == "residual" and block_id > 0:
                cmd_append, lin_comb_cnt = self.add_residual(
                    cmd_append, block_id, out_lst, lin_comb_cnt
                )
                current_inp = self.out_var[0] + "_" + str(block_id) + "_res"

            # Adding dense connection:
            elif self.add_connections == "dense" and block_id > 0:
                cmd_append, lin_comb_cnt = self.add_dense(
                    cmd_append, block_id, out_lst, lin_comb_cnt
                )
                current_inp = self.out_var[0] + "_" + str(block_id) + "_dense"

            else:
                current_inp = self.out_var[0] + "_" + str(block_id)

            # Appending command
            self.config_proc_replicate["computations"] = (
                self.config_proc_replicate["computations"] + cmd_append + "\n"
            )

        # Managing dense connections
        if self.add_connections == "dense":
            self.config_proc_replicate["computations"] = (
                self.config_proc_replicate["computations"]
                + str(self.out_var[0])
                + "="
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_dense"
            )

        # Managing residual connections
        elif self.add_connections == "residual":
            self.config_proc_replicate["computations"] = (
                self.config_proc_replicate["computations"]
                + str(self.out_var[0])
                + "="
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_res"
            )

        # Managing skip connections
        elif self.add_connections == "skip":
            lin_comb_cnt = self.add_skip(block_id, out_lst, lin_comb_cnt)
            self.config_proc_replicate["computations"] = (
                self.config_proc_replicate["computations"]
                + str(self.out_var[0])
                + "="
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_skip"
            )

        # Managing standard case (no shortcut connections)
        else:
            self.config_proc_replicate["computations"] = (
                self.config_proc_replicate["computations"]
                + str(self.out_var[0])
                + "="
                + self.out_var[0]
                + "_"
                + str(block_id)
            )

        self.config_proc_replicate[
            "computations"
        ] = self.config_proc_replicate["computations"].split("\n")

    def add_residual(self, cmd_append, block_id, out_lst, lin_comb_cnt):
        """
         ---------------------------------------------------------------------
         core.execute_computations.add_residual
         (author: Mirco Ravanelli)

         Description: This function creates residual connnections when
                      replicating the basic computation block.

         ---------------------------------------------------------------------
         """

        # Managing difference combination
        if self.connection_merge == "diff":
            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_res="
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "-"
                + out_lst[-2]
            )

        # Managing sum combination
        if self.connection_merge == "sum":
            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_res="
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "+"
                + out_lst[-2]
            )

        # Managing average combination
        if self.connection_merge == "average":
            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_res=("
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "+"
                + out_lst[-2]
                + ")/2"
            )

        # Managing concat combination
        if self.connection_merge == "concat":
            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_res=torch.cat(["
                + self.out_var[0]
                + "_"
                + str(block_id)
                + ","
                + out_lst[-2]
                + "],dim=1)"
            )

        # Managing linear combination
        if self.connection_merge == "linear_comb":

            # Adding linear_comb function in the function pool
            lin_comb_id = "linear_comb_" + str(lin_comb_cnt)
            lin_comb_cfg = {"class_name": "neural_networks.linear_combination"}
            self.config_proc_replicate["functions"][lin_comb_id] = lin_comb_cfg

            # Adding computation
            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + '_res=self.run_function("linear_comb_'
                + str(lin_comb_cnt)
                + '",'
                + out_lst[-2]
                + ","
                + self.out_var[0]
                + "_"
                + str(block_id)
                + ")"
            )

            # Updating linear combination counter
            lin_comb_cnt = lin_comb_cnt + 1

        return cmd_append, lin_comb_cnt

    def add_dense(self, cmd_append, block_id, out_lst, lin_comb_cnt):
        """
         ---------------------------------------------------------------------
         core.execute_computations.add_dense
         (author: Mirco Ravanelli)

         Description: This function creates dense connnections when
                      replicating the basic computation block.

         ---------------------------------------------------------------------
         """

        # Managing difference combination
        if self.connection_merge == "diff":
            cmd_dense = ""
            for out in out_lst:
                cmd_dense = cmd_dense + out + "-"

            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_dense="
                + cmd_dense[0:-1]
            )

        # Managing sum combination
        if self.connection_merge == "sum":
            cmd_dense = ""
            for out in out_lst:
                cmd_dense = cmd_dense + out + "+"

            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_dense="
                + cmd_dense[0:-1]
            )

        # Managing average combination
        if self.connection_merge == "average":
            cmd_dense = ""
            for out in out_lst:
                cmd_dense = cmd_dense + out + "/" + str(len(out_lst)) + "+"

            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_dense="
                + cmd_dense[0:-1]
            )

        # Managing concat combination
        if self.connection_merge == "concat":
            cmd_dense = "torch.cat(["
            for out in out_lst:
                cmd_dense = cmd_dense + out + ","

            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_dense="
                + cmd_dense[0:-1]
                + "],dim=1)\n"
            )

        # Managing linear combination
        if self.connection_merge == "linear_comb":

            # adding linear combination function
            lin_comb_id = "linear_comb_" + str(lin_comb_cnt)
            lin_comb_cfg = {"class_name": "neural_networks.linear_combination"}
            self.config_proc_replicate["functions"][lin_comb_id] = lin_comb_cfg

            cmd_dense = ""

            # Linear combination of all the block outputs
            for out in out_lst:

                # adding computation
                cmd_dense = cmd_dense + out + ","

            cmd_append = (
                "\n"
                + cmd_append
                + self.out_var[0]
                + "_"
                + str(block_id)
                + '_dense=self.run_function("linear_comb_'
                + str(lin_comb_cnt)
                + '",'
                + cmd_dense[:-1]
                + ")\n"
            )
            lin_comb_cnt = lin_comb_cnt + 1

        return cmd_append, lin_comb_cnt

    def add_skip(self, block_id, out_lst, lin_comb_cnt):
        """
         ---------------------------------------------------------------------
         core.execute_computations.add_skip
         (author: Mirco Ravanelli)

         Description: This function creates skip connnections when
                      replicating the basic computation block.

         ---------------------------------------------------------------------
        """

        # Managing sum combination
        if self.connection_merge == "sum":
            cmd_skip = ""
            for out in out_lst:
                cmd_skip = cmd_skip + out + "+"

            self.config_proc_replicate["computations"] = (
                "\n"
                + self.config_proc_replicate["computations"]
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_skip="
                + cmd_skip[0:-1]
                + "\n"
            )

        # Managing difference combination
        if self.connection_merge == "diff":
            cmd_skip = ""
            for out in out_lst:
                cmd_skip = cmd_skip + out + "-"

            self.config_proc_replicate["computations"] = (
                "\n"
                + self.config_proc_replicate["computations"]
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_skip="
                + cmd_skip[0:-1]
                + "\n"
            )

        # Managing average combination
        if self.connection_merge == "average":
            cmd_skip = ""
            for out in out_lst:
                cmd_skip = cmd_skip + out + "/" + str(len(out_lst)) + "+"

            self.config_proc_replicate["computations"] = (
                "\n"
                + self.config_proc_replicate["computations"]
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_skip="
                + cmd_skip[0:-1]
                + "\n"
            )

        # Managing concat combination
        if self.connection_merge == "concat":
            cmd_skip = "torch.cat(["
            for out in out_lst:
                cmd_skip = cmd_skip + out + ","

            self.config_proc_replicate["computations"] = (
                "\n"
                + self.config_proc_replicate["computations"]
                + self.out_var[0]
                + "_"
                + str(block_id)
                + "_skip="
                + cmd_skip[0:-1]
                + "],dim=1)\n"
            )

        # Managing linear combination
        if self.connection_merge == "linear_comb":

            # Adding linear combination function
            lin_comb_id = "linear_comb_" + str(lin_comb_cnt)
            lin_comb_cfg = {"class_name": "neural_networks.linear_combination"}
            self.config_proc_replicate["functions"][lin_comb_id] = lin_comb_cfg

            cmd_skip = ""

            # Combining all the outputs of the blocks
            for out in out_lst:
                # Adding computation
                cmd_skip = cmd_skip + out + ","

            self.config_proc_replicate["computations"] = (
                "\n"
                + self.config_proc_replicate["computations"]
                + self.out_var[0]
                + "_"
                + str(block_id)
                + '_skip=self.run_function("linear_comb_'
                + str(lin_comb_cnt)
                + '",'
                + cmd_skip[:-1]
                + ")\n"
            )
            lin_comb_cnt = lin_comb_cnt + 1

        return lin_comb_cnt

    def create_replace_dict(self,):
        """
         ---------------------------------------------------------------------
         core.execute_computations.create_replace_dict
         (author: Mirco Ravanelli)

         Description: This function analysize the replicate_with field and
                      creates a dictionary with the changes to to at each
                      replica.

         Input:       -None


         Output:      - None
                         the dictionary is written in self.replace_dict

         Example:    from speechbrain.core import execute_computations

                     cfg_file='cfg/minimal_examples/neural_networks/spk_id/\
                     spk_id_example.cfg'

                     # Definition of the exec_config dictionary
                     exec_config={'class_name':'core.execute_computations', \
                                  'cfg_file': cfg_file}

                     # Initialization of the class
                     computations=execute_computations(exec_config)

                     # set the number of replicas
                     computations.replicate_with='conv2d,kernel_size=\
                         5:9*2,9:5 conv2d,out_channels=2*3'

                     # computing the replace dict
                     computations.create_replace_dict()

                     # replace_dict
                     print(computations.replace_dict)

         ---------------------------------------------------------------------
         """

        # Initialization of the dictionary
        self.replace_dict = {}

        # Creating the replace _dict
        if self.replicate_with is not None:

            # Separate the replicate fields
            replicate_with_fields = self.replicate_with.split(" ")

            # Processing all the entries
            for repl_str in replicate_with_fields:

                # Getting the field to change
                field = repl_str.split("=")[0].replace(",", "_")

                self.replace_dict[field] = []

                # Getting the values
                values = repl_str.split("=")[1].split(",")

                # Processing all the values
                for value in values:

                    val_append = [value]

                    # '*' indicates the number of replica to get with
                    # the corresponding value
                    if "*" in value:
                        val = value.split("*")[0]
                        multiplier = int(value.split("*")[1])

                        # ':' is use to set up the value to replace
                        if ":" in val:
                            val = val.replace(":", ",")

                        val_append = [val] * multiplier

                    else:

                        if ":" in value:
                            val_append = [value.replace(":", ",")]

                    # Appending value to the replace dictionary
                    self.replace_dict[field] = (
                        self.replace_dict[field] + val_append
                    )

    def detect_cmd_to_replicate(self, cmd_str):
        """
         ---------------------------------------------------------------------
         core.execute_computations.detect_cmd_to_replicate
         (author: Mirco Ravanelli)

         Description: This function analyzes the command string and detects
                      automatically the block of computations to replicate.

         Input:       - cmd_str (type: string, mandatory)
                         it is a string contaning the basic computation
                         to replicate.


         Output:      - cmd_to_replicate,current_inp (type: str)
                         it is the part of the command to replicate
         ---------------------------------------------------------------------
         """
        self.config_proc_replicate["computations"] = ""

        cmd_to_replicate = ""

        first_comp = True

        # Loop over all the command lines
        for cmd_id, cmd in enumerate(cmd_str.split("\n")):

            # Append get_input_var
            if "get_input_var()" in cmd:
                self.config_proc_replicate["computations"] = cmd + "\n"
            else:

                if first_comp:

                    # Check the input of this computation block
                    current_inp = cmd[
                        cmd.find('",') + 2: cmd.find(")")
                    ].split(",")[0]

                    # replace input with the tag $current_inp
                    # this operation is useful to later detect the input
                    # of the computation block. This must be done only
                    # for hte first computation line (i.e, the one take
                    # takes the input variable)
                    cmd = cmd.replace("," + current_inp, ",$current_inp")

                    first_comp = False

                if cmd_id == len(cmd_str.split("\n")) - 1:

                    current_out = cmd.split("=")[0].split(",")[0]

                    out_vars = cmd.split("=")[0]
                    cmd = cmd.split("=")[1:]

                    # Tagging the current output as $current_out
                    out_vars = out_vars.replace(current_out, "$current_out")

                    cmd = out_vars + "=" + "=".join(cmd)

                # Appending command
                cmd_to_replicate = cmd_to_replicate + cmd + "\n"

        return cmd_to_replicate, current_inp

    def replicate_functions(self):
        """
         ---------------------------------------------------------------------
         core.execute_computations.replicate_functions
         (author: Mirco Ravanelli)

         Description: This function check for all the functions defined in
                      the config file and replicates them.
         -----
        """

        self.config_proc_replicate["functions"] = {}

        # Replicate the functions self.replicate times
        for block_id in range(self.replicate):

            # Loop over all the function defined in the config file
            for funct_name in self.config_proc["functions"]:

                # Replicate the function with a different name
                # (i.e, funct_name_block_id)
                funct_rep = funct_name + "_" + str(block_id)
                self.config_proc_replicate["functions"][
                    funct_rep
                ] = self.config_proc["functions"][funct_name].copy()

                # Replace parameters according to replace_with
                for field in self.config_proc_replicate["functions"][
                    funct_rep
                ]:
                    if funct_name + "_" + field in self.replace_dict:
                        self.config_proc_replicate["functions"][funct_rep][
                            field
                        ] = self.replace_dict[funct_name + "_" + field][
                            block_id
                        ]
