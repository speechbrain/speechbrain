"""
 -----------------------------------------------------------------------------
 merge_scp.py (author: Mirco Ravanelli)

 Description: This script merges two scp file

 Input:       scp_lst (list, mandatory):
                 it is a list containing the scp files (passed through command
                 line)

 Output:      scp_out (str, mandatory)
                 it is the path where the combined scp file will be saved.
                 By th default it is the last argument of the command line

 Example:     python merge_scp.py \
              samples/audio_samples/scp_example.scp \
              exp/read_write_mininal_noise_scp/noisy_wav/scp.scp \
              out.scp
 -----------------------------------------------------------------------------
"""

import os
import sys
from utils import scp_to_dict, dict_to_str, logger_write


if __name__ == '__main__':

    scp_lst = sys.argv[1:]

    # Check if the input scp files exist
    for inp_scp_file in scp_lst[:-1]:
        if not(os.path.isfile(inp_scp_file)):
            err_msg = 'the file "%s" does not exist' % (inp_scp_file)
            logger_write(err_msg)

    # Output scp file
    fout = open(scp_lst[-1], "w")

    # Converting scp to data dicts
    data_dicts = []

    for scp_file in scp_lst[:-1]:
        data_dict = scp_to_dict(scp_file)
        data_dicts.append(data_dict)

    # Updating elements
    for snt_id in data_dicts[0]:

        for data_dict in data_dicts[1:]:

            # update the snts with the new features
            if snt_id in data_dict:
                data_dicts[0][snt_id].update(data_dict[snt_id])

            scp_line_out = dict_to_str(data_dicts[0][snt_id])
            fout.write(scp_line_out)

    fout.close()
