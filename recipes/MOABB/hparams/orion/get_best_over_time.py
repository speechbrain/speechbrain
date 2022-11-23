"""
This code finds the best hparams obtained after the specified checkpoints in minutes.
The assumption is that you already ran the hparam tuning phase with Orion.
Make sure the orion database is added in your bash (e.g. type export ORION_DB_ADDRESS=/network/scratch/r/ravanelm/tpe_EEGNet_BNCI2014001_original.pkl)
Set the variables below for your experiment (default is EEGNet_BNCI2014001).
The output is the set of commands to run.

For instance, one of the output can look like this 

./run_experiments_seed_variability.sh hparams/EEGNet_BNCI2014001_seed_variability.yaml /localscratch/eeg_data results/EEGNet_BNCI2014001_tpe_seach_EEGNet_BNCI2014001_original_exp_1_minute_3 9 2 random_seed 5 acc valid_metrics.pkl false true --avg_models=6 --batch_size_exponent=6 --cnn_septemporal_kernelsize_=9 --cnn_septemporal_point_kernels_ratio_=3 --cnn_septemporal_pool=5 --cnn_spatial_depth_multiplier=3 --cnn_temporal_kernels=20 --cnn_temporal_kernelsize=51 --dropout=0.4441 --fmax=42.1 --fmin=1.3 --idx_combination_augmentations=11 --lr=0.005 --n_steps_channel_selection=1 --number_of_epochs=292 --repeat_augment=0 --tmax=1.3 
"""

from orion.client import get_experiment
import os
import sys


def get_status(expriment_name, out_file_name):
    cmd = "orion status -n " + expriment_name + " --all >" + out_file_name
    os.system(cmd)


def track_best_over_time(status_file, out_file_name):

    # minutes (it takes 12 hours to do 250 experiments)
    avg_duration_exp = 2.88
    time = 0
    cnt = 0

    with open(status_file) as file:
        lines = [line.rstrip() for line in file]

    out_lines = []
    # process lines
    for line in lines:
        if "completed" in line:
            time = time + avg_duration_exp
            cnt = cnt + 1
            trial_id = line.split(" ")[0]
            err = float(line.split(" ")[-1])
            if cnt == 1:
                err_best = err
                id_best = trial_id
            else:
                if err < err_best:
                    err_best = err
                    id_best = trial_id

            line_out = (
                str(cnt)
                + " "
                + str(round(time))
                + " "
                + str(err)
                + " "
                + trial_id
                + " "
                + str(err_best)
                + " "
                + id_best
                + "\n"
            )
            out_lines.append(line_out)

    with open(out_file_name, "w") as f:
        f.writelines(out_lines)
    return out_lines


def track_best_over_selecred_time(best_list, check_best_after_minutes, out_file_name):
    id_check = 0
    out_lines = []
    for line in best_list:
        time = float(line.split(" ")[1])
        if time >= check_best_after_minutes[id_check]:
            out_lines.append(line)
            id_check = id_check + 1

    out_file_name = expriment_name + "_best_over_selected_time.txt"
    with open(out_file_name, "w") as f:
        f.writelines(out_lines)
    return out_lines


def get_hparams(expriment_name, best_list, out_file_name):
    hparam_list = []
    experiment = get_experiment(expriment_name)
    for best in best_list:
        best_trial = best.split(" ")[-1].strip()
        best_hpar = experiment.get_trial(uid=best_trial).params
        hparam_list.append(best_hpar)
    with open(out_file_name, "w") as f:
        f.writelines(str(hparam_list))
    return hparam_list


def get_cmd(
    hparams,
    data_folder,
    output_folder,
    nsbj,
    nsess,
    nruns,
    seed,
    eval_metric,
    metric_file,
    do_leave_one_subject_out,
    do_leave_one_session_out,
    out_file_name,
):
    cmd_list = []
    for hparms, best_line in zip(hparam_list, best_over_selected_time_list):
        exp_ind = best_line.split(" ")[0]
        exp_minute = best_line.split(" ")[1]
        output_folder_exp = output_folder + "_exp_" + exp_ind + "_minute_" + exp_minute
        cmd_base = (
            "./run_experiments_seed_variability.sh hparams/EEGNet_BNCI2014001_seed_variability.yaml "
            + data_folder
            + " "
            + output_folder_exp
            + " "
            + str(nsbj)
            + " "
            + str(nsess)
            + " "
            + seed
            + " "
            + str(nruns)
            + " "
            + eval_metric
            + " "
            + metric_file
            + " "
            + str(do_leave_one_subject_out)
            + " "
            + str(do_leave_one_session_out)
            + " "
        )
        cmd_flags = ""
        for key in hparms.keys():
            cmd_flags = (
                cmd_flags + "--" +
                key.replace("/", "") + "=" + str(hparms[key]) + " "
            )
        final_cmd = cmd_base + cmd_flags
        print(final_cmd)
        cmd_list.append(final_cmd)
    with open(out_file_name, "w") as f:
        f.writelines(cmd_list)


# as reported in orion-list (without the final 'v1')
expriment_name = "tpe_seach_EEGNet_BNCI2014001_original"
status_file = expriment_name + "_status_all.txt"
get_status(expriment_name, status_file)
best_over_time_file = expriment_name + "_best_over_time.txt"
best_list = track_best_over_time(status_file, best_over_time_file)

check_best_after_minutes = [0, 60, 120, 240, 480, 720]

best_over_selected_time_file = expriment_name + "_best_over_selected_time.txt"
best_over_selected_time_list = track_best_over_selecred_time(
    best_list, check_best_after_minutes, best_over_selected_time_file
)

best_over_selected_time_file = expriment_name + \
    "_best_hparams_over_selected_time.txt"
hparam_list = get_hparams(
    expriment_name, best_over_selected_time_list, best_over_selected_time_file
)


hparams = "hparams/EEGNet_BNCI2014001_seed_variability.yaml"
data_folder = "/localscratch/eeg_data"
output_folder = "results/EEGNet_BNCI2014001_" + expriment_name
nsbj = 9
nsess = 2
nruns = 5
seed = "random_seed"
eval_metric = "acc"
metric_file = "valid_metrics.pkl"
do_leave_one_subject_out = "false"
do_leave_one_session_out = "true"

cmd_to_run = expriment_name + "_best_commands.sh"
get_cmd(
    hparams,
    data_folder,
    output_folder,
    nsbj,
    nsess,
    nruns,
    seed,
    eval_metric,
    metric_file,
    do_leave_one_subject_out,
    do_leave_one_session_out,
    cmd_to_run,
)
