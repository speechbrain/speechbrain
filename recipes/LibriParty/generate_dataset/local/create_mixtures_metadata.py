import glob
import soundfile as sf
import numpy as np
from pathlib import Path
import yaml
import json
import os
from tqdm import tqdm


def _read_metadata(file_path, configs):
    meta = sf.SoundFile(file_path)
    if meta.channels > 1:
        channel = np.random.randint(0, meta.channels - 1)
    else:
        channel = 0
    assert (
            meta.samplerate == configs["samplerate"]
    ), "file samplerate is different from the one specified"

    return meta, channel


def create_metadata(
    configs,
    utterances_dict,
    words_dict,
    rir_list,
    impulsive_noises_list=None,
    background_noises_list=None,
):

    dataset_metadata = {}
    for n_sess in tqdm(range(configs["n_sessions"])):
        # we sample randomly n_speakers ids
        c_speakers = np.random.choice(
            list(utterances_dict.keys()), configs["n_speakers"]
        )
        # we select all utterances from selected speakers
        c_utts = [utterances_dict[spk_id] for spk_id in c_speakers]

        activity = {}
        for spk in c_speakers:
            activity[spk] = []

        tot_length = 0
        min_spk_lvl = np.inf
        for i in range(len(c_speakers)):
            c_spk = c_speakers[i]
            spk_utts = c_utts[i]
            np.random.shuffle(spk_utts)  # random shuffle
            intervals = np.random.exponential(
                configs["interval_factor_speech"], len(spk_utts)
            )  # we use same technique as in EEND repo for intervals distribution

            cursor = 0
            for j, wait in enumerate(intervals):
                meta, channel = _read_metadata(spk_utts[j], configs)
                c_rir = np.random.choice(rir_list, 1)[0]
                # check if the rir is monaural
                meta_rir, rir_channel = _read_metadata(c_rir, configs)
                length = len(meta) / meta.samplerate
                id_utt = Path(spk_utts[j]).stem
                cursor += wait
                if cursor + length > configs["max_length"]:
                    break

                lvl = np.clip(
                    np.random.normal(
                        configs["speech_lvl_mean"], configs["speech_lvl_var"]
                    ),
                    configs["speech_lvl_min"],
                    configs["speech_lvl_max"],
                )
                min_spk_lvl = min(lvl, min_spk_lvl)
                # we save to metadata only relative paths
                activity[c_spk].append(
                    {
                        "start": cursor,
                        "stop": cursor + length,
                        "words": words_dict[id_utt],
                        "rir": str(
                            Path(c_rir).relative_to(configs["rirs_root"])
                        ),
                        "utt_id": id_utt,
                        "file": str(
                            Path(spk_utts[j]).relative_to(
                                configs["librispeech_root"]
                            )
                        ),
                        "lvl": lvl,
                        "channel": channel,
                        "rir_channel": rir_channel,
                    }
                )
                tot_length = max(cursor + length, tot_length)
                cursor = cursor + length

        # we add also impulsive noises as it were a speaker
        if impulsive_noises_list:
            activity["noises"] = []
            # sampling intervals for impulsive noises.
            intervals = np.random.exponential(
                configs["interval_factor_noises"], configs["n_imp_noises"]
            )
            cursor = 0
            for j, wait in enumerate(intervals):
                # we sample with replacement an impulsive noise.
                c_noise = np.random.choice(impulsive_noises_list, 1)[0]
                meta, channel = _read_metadata(c_noise, configs)
                c_rir = np.random.choice(rir_list, 1)[0]
                # we reverberate it.
                meta_rir, rir_channel = _read_metadata(c_rir, configs)
                length = len(meta) / meta.samplerate
                cursor += wait
                if cursor + length > configs["max_length"]:
                    break
                lvl = np.clip(
                    np.random.normal(
                        configs["imp_lvl_mean"], configs["imp_lvl_var"]
                    ),
                    configs["imp_lvl_min"],
                    min(min_spk_lvl + configs["imp_lvl_rel_max"], 0),
                )

                activity["noises"].append(
                    {
                        "start": cursor,
                        "stop": cursor + length,
                        "rir": str(
                            Path(c_rir).relative_to(configs["rirs_root"])
                        ),
                        "file": str(
                            Path(c_noise).relative_to(
                                configs["impulsive_noises_root"]
                            )
                        ),
                        "lvl": lvl,
                        "channel": channel,
                        "rir_channel": rir_channel,
                    }
                )
                tot_length = max(tot_length, cursor + length)
                cursor += length

        if background_noises_list:
            # we add also background noise.
            lvl = np.random.randint(
                configs["background_lvl_min"],
                min(min_spk_lvl + configs["background_lvl_rel_max"], 0),
            )
            # we scale the level but do not reverberate.
            background = np.random.choice(background_noises_list, 1)[0]
            meta, channel = _read_metadata(background, configs)
            assert (
                len(meta) >= configs["max_length"] * configs["samplerate"]
            ), "background noise files should be >= max_length in duration"
            offset = 0
            if len(meta) > configs["max_length"] * configs["samplerate"]:
                offset = np.random.randint(
                    0,
                    len(meta)
                    - int(configs["max_length"] * configs["samplerate"]),
                )

            activity["background"] = {
                "start": 0,
                "stop": tot_length,
                "file": str(
                    Path(background).relative_to(configs["backgrounds_root"])
                ),
                "lvl": lvl,
                "orig_start": offset,
                "orig_stop": offset + int(tot_length * configs["samplerate"]),
                "channel": channel,
            }
        else:
            # we use as background gaussian noise
            lvl = np.random.randint(
                configs["background_lvl_min"],
                min(min_spk_lvl + configs["background_lvl_rel_max"], 0),
            )
            activity["background"] = {
                "start": 0,
                "stop": tot_length,
                "file": None,
                "lvl": lvl,
                "orig_start": None,
                "orig_stop": None,
                "channel": None,
            }

        dataset_metadata["session_{}".format(n_sess)] = activity

    with open(os.path.join(configs["out_path"], "metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)


if __name__ == "__main__":
    import sys
    import speechbrain as sb

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from create_mixtures_metadata import create_metadata  # noqa E402

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])

    # we load parameters
    with open(params_file, "r") as f:
        params = yaml.load(f)

    os.makedirs(params["out_path"], exist_ok=True)
    # parsing librispeech
    # step 1:  we construct a dictionary with speakers ids
    utterances = []
    txt_files = []
    for libri_dir in params["librispeech_folders"]:
        utterances.extend(
            glob.glob(os.path.join(libri_dir, "**/*.flac"), recursive=True)
        )
        txt_files.extend(
            glob.glob(os.path.join(libri_dir, "**/*trans.txt"), recursive=True)
        )

    # step 2: we then build an hashtable for words for each utterance
    words_dict = {}
    for trans in txt_files:
        with open(trans, "r") as f:
            for line in f:
                splitted = line.split(" ")
                utt_id = splitted[0]
                words = " ".join(splitted[1:])
                words_dict[utt_id] = words.strip("\n")

    # step 3: we build an hashtable for also speakers
    speakers = {}
    for u in utterances:
        spk_id = Path(u).parent.parent.stem
        if spk_id not in speakers:
            speakers[spk_id] = [u]
        else:
            speakers[spk_id].append(u)
    ### done parsing librispeech ####
    # we now parse rirs, noises and backgrounds
    rirs_list = []
    if params["rirs_folders"]:
        for rir_folder in params["rirs_folders"]:
            rirs_list.extend(
                glob.glob(os.path.join(rir_folder), recursive=True)
            )

    # we parse impulsive noises
    impulsive_noises = []
    if params["impulsive_noises_folders"]:
        for imp_folder in params["impulsive_noises_folders"]:
            impulsive_noises.extend(
                glob.glob(os.path.join(imp_folder), recursive=True)
            )
    # we parse background noise
    backgrounds = []
    if params["background_noises_folders"]:
        for back in params["background_noises_folders"]:
            backgrounds.extend(glob.glob(back, recursive=True))

    create_metadata(
        params,
        speakers,
        words_dict,
        rirs_list,
        impulsive_noises_list=impulsive_noises,
        background_noises_list=backgrounds,
    )
