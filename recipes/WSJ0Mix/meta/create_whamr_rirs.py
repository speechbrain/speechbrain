import os
import numpy as np
import pandas as pd
from constants import SAMPLERATE
import argparse
import torchaudio

from wham_room import WhamRoom
from scipy.signal import resample_poly
import torch

FILELIST_STUB = os.path.join("data", "mix_2_spk_filenames_{}.csv")

SINGLE_DIR = "mix_single"
BOTH_DIR = "mix_both"
CLEAN_DIR = "mix_clean"
S1_DIR = "s1"
S2_DIR = "s2"
NOISE_DIR = "noise"

RIR_DIR = "rirs"
SUFFIXES = ["_anechoic", "_reverb"]

MONO = True  # Generate mono audio, change to false for stereo audio
SPLITS = ["tr", "cv", "tt"]
SAMPLE_RATES = ["8k"]  # Remove element from this list to generate less data
DATA_LEN = ["min"]  # Remove element from this list to generate less data


def create_wham(wsj_root, wham_noise_path, output_root, args):
    # LEFT_CH_IND = 0
    # if MONO:
    #    ch_ind = LEFT_CH_IND
    # else:
    #    ch_ind = [0, 1]

    scaling_npz_stub = os.path.join(
        wham_noise_path, "metadata", "scaling_{}.npz"
    )
    reverb_param_stub = os.path.join("data", "reverb_params_{}.csv")

    for splt in SPLITS:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)

        scaling_npz_path = scaling_npz_stub.format(splt)
        scaling_npz = np.load(scaling_npz_path, allow_pickle=True)

        # noise_path = os.path.join(wham_noise_path, splt)

        reverb_param_path = reverb_param_stub.format(splt)
        reverb_param_df = pd.read_csv(reverb_param_path)

        for wav_dir in ["wav" + sr for sr in SAMPLE_RATES]:
            for datalen_dir in DATA_LEN:
                output_path = os.path.join(
                    output_root, wav_dir, datalen_dir, splt
                )
                os.makedirs(os.path.join(output_path, RIR_DIR), exist_ok=True)

        utt_ids = scaling_npz["utterance_id"]
        # start_samp_16k = scaling_npz["speech_start_sample_16k"]

        for i_utt, output_name in enumerate(utt_ids):
            utt_row = reverb_param_df[
                reverb_param_df["utterance_id"] == output_name
            ]
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

            for sr_i, sr_dir in enumerate(SAMPLE_RATES):
                #     wav_dir = 'wav' + sr_dir
                if sr_dir == "8k":
                    sr = 8000
                    # downsample = True
                else:
                    sr = SAMPLERATE
                    # downsample = False

                for datalen_dir in DATA_LEN:
                    output_path = os.path.join(
                        output_root, wav_dir, datalen_dir, splt
                    )

                    hs = []
                    for i, mics in enumerate(rir):
                        sources = []
                        for j, source in enumerate(mics):
                            h = resample_poly(source, sr, 16000)
                            h_torch = torch.from_numpy(h).float()
                            sources.append(h_torch)
                            if args.save_style == "random":

                                torchaudio.save(
                                    os.path.join(
                                        args.random_save_path,
                                        "{}_{}_".format(i, j) + output_name,
                                    ),
                                    source,
                                    8000,
                                )

                        hs.append(sources)

                    if args.save_style == "sorted":
                        path = os.path.join(
                            output_path, RIR_DIR, output_name + ".t"
                        )
                        torch.save(hs, path)

            if (i_utt + 1) % 500 == 0:
                print(
                    "Completed {} of {} utterances".format(
                        i_utt + 1, len(wsjmix_df)
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.",
    )
    parser.add_argument(
        "--wsj0-root",
        type=str,
        required=True,
        help="Path to the folder containing wsj0/",
    )
    parser.add_argument(
        "--wham-noise-root",
        type=str,
        required=True,
        help="Path to the downloaded and unzipped wham folder containing metadata/",
    )
    parser.add_argument(
        "--save_style",
        type=str,
        default="random",
        help="either random or sorted. ",
    )
    parser.add_argument(
        "--random_save_path",
        type=str,
        required=True,
        help="The path for saving the rirs for random augmentation style",
    )

    args = parser.parse_args()
    create_wham(args.wsj0_root, args.wham_noise_root, args.output_dir, args)
