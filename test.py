"""
 -----------------------------------------------------------------------------
 test.py (author: Mirco Ravanelli)

 Description: This is the a script that performs some basic tests on the
              code of the SpeechBrain toolkit. Please, run this code before
              asking for a pull request.

 Input:       None

 Output:      None

 Example:     python test.py
 -----------------------------------------------------------------------------
"""

import sys
import pycodestyle
from utils import create_exec_config
from core import execute_computations

# List of config files to run:
cfg_lst = ['cfg/minimal_examples/data_reading/read_data_example.cfg',
           'cfg/minimal_examples/data_reading/read_data_example2.cfg',
           'cfg/minimal_examples/data_reading/read_write_data.cfg',
           'cfg/minimal_examples/data_reading/loop_example.cfg',
           'cfg/minimal_examples/basic_processing/'
           'minimal_processing_read_write_example_noise.cfg',
           'cfg/minimal_examples/basic_processing/'
           'minimal_processing_read_write_example_noise_parallel.cfg',
           'cfg/minimal_examples/features/compute_stft_example.cfg',
           'cfg/minimal_examples/features/'
           'compute_spectrogram_example.cfg',
           'cfg/minimal_examples/features/compute_fbanks_example.cfg',
           'cfg/minimal_examples/features/compute_mfccs_example.cfg',
           'cfg/minimal_examples/features/compute_mfccs_example2.cfg',
           'cfg/minimal_examples/features/compute_mfccs_example3.cfg',
           'cfg/minimal_examples/multichannel/'
           'compute_stft_multichannel_example.cfg',
           'cfg/minimal_examples/multichannel/'
           'compute_spectrogram_multichannel_example.cfg',
           'cfg/minimal_examples/multichannel/'
           'compute_fbanks_multichannel_example.cfg']


# List of files to check:
check_lst = ['speechbrain.py', 'core.py', 'data_processing.py', 'data_io.py',
             'data_preparation.py', 'utils.py']


# Running examples in config files
for cfg_file in cfg_lst:

    print('checking %s' % cfg_file)
    # Creating config dict for executing computations
    exec_config = create_exec_config(cfg_file, '')

    exec_config.update({"cfg_change": '--global,verbosity=0'})

    # Initializing the execute computation class
    computations = execute_computations(exec_config)

    # Executing the computations specified in the config file
    computations([])

# Checking PEP8 consistency
print('check PEP8 consistency:')

for file in check_lst:

    # Checking if the file is compliant with PEP8
    fchecker = pycodestyle.Checker(file, show_source=True)
    file_errors = fchecker.check_all()

    print("Found %s errors (and warnings) in %s" % (file_errors, file))

    if file_errors > 0:
        sys.exit(0)

print('Test finished without errors!')
