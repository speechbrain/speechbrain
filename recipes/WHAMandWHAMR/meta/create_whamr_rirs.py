"""
Adapted from the original WHAMR script to obtain the Room Impulse ResponsesRoom Impulse Responses

Authors
    * Cem Subakan 2021
"""

import argparse
import os

import pandas as pd
import torch
import torchaudio
from scipy.signal import resample_poly
from tqdm import tqdm
from wham_room import WhamRoom

from speechbrain.utils.fetching import fetch


def create_rirs(output_dir, sr=8000):
    """
    This function creates the room impulse responses from the WHAMR! dataset
    The implementation is based on the scripts from http://wham.whisper.ai/

    Arguments:
    ------
    output_dir (str) : directory for saving the RIRs
    sr (int) : sampling rate with which we save

    """

    os.makedirs(output_dir)

    metafilesdir = os.path.dirname(os.path.realpath(__file__))
    filelist = [
        "mix_2_spk_filenames_tr.csv",
        "mix_2_spk_filenames_cv.csv",
        "mix_2_spk_filenames_tt.csv",
        "reverb_params_tr.csv",
        "reverb_params_cv.csv",
        "reverb_params_tt.csv",
    ]

    savedir = os.path.join(metafilesdir, "data")
    for fl in filelist:
        if not os.path.exists(os.path.join(savedir, fl)):
            fetch(
                "metadata/" + fl,
                "speechbrain/sepformer-whamr",
                savedir=savedir,
                save_filename=fl,
            )

    FILELIST_STUB = os.path.join(metafilesdir, "data", "mix_2_spk_filenames_{}.csv")

    SPLITS = ["tr"]

    reverb_param_stub = os.path.join(metafilesdir, "data", "reverb_params_{}.csv")

    for split in SPLITS:
        wsjmix_path = FILELIST_STUB.format(split)
        wsjmix_df = pd.read_csv(wsjmix_path)

        reverb_param_path = reverb_param_stub.format(split)
        reverb_param_df = pd.read_csv(reverb_param_path)

        utt_ids = wsjmix_df.output_filename.values

        for output_name in tqdm(utt_ids):
            utt_row = reverb_param_df[reverb_param_df["utterance_id"] == output_name]
            room = WhamRoom(
                [
                    utt_row["room_x"].iloc[0],
                    utt_row["room_y"].iloc[0],
                    utt_row["room_z"].iloc[0],
                ],
                [
                    [
                        utt_row["micL_x"].iloc[0],
                        utt_row["micL_y"].iloc[0],
                        utt_row["mic_z"].iloc[0],
                    ],
                    [
                        utt_row["micR_x"].iloc[0],
                        utt_row["micR_y"].iloc[0],
                        utt_row["mic_z"].iloc[0],
                    ],
                ],
                [
                    utt_row["s1_x"].iloc[0],
                    utt_row["s1_y"].iloc[0],
                    utt_row["s1_z"].iloc[0],
                ],
                [
                    utt_row["s2_x"].iloc[0],
                    utt_row["s2_y"].iloc[0],
                    utt_row["s2_z"].iloc[0],
                ],
                utt_row["T60"].iloc[0],
            )
            room.generate_rirs()

            rir = room.rir_reverberant

            for i, mics in enumerate(rir):
                for j, source in enumerate(mics):
                    h = resample_poly(source, sr, 16000)
                    h_torch = torch.from_numpy(h).float().unsqueeze(0)

                    torchaudio.save(
                        os.path.join(output_dir, "{}_{}_".format(i, j) + output_name),
                        h_torch,
                        sr,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The output directory for saving the rirs for random augmentation style",
    )

    args = parser.parse_args()
    create_rirs(args.output_dir)
