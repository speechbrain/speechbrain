import os
import logging
from speechbrain.dataio.dataio import read_audio

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


def prepare_FSC(data_folder, save_folder, skip_prep=False):
    """
    This function prepares the Fluent Speech Commands dataset.

    data_folder : path to dataset.
    save_folder: folder where the manifest files will be stored.
    skip_prep: If True, skip data preparation

    """
    if skip_prep:
        return

    splits = [
        "train",
        "valid",
        "test",
    ]
    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        ID = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []

        spk_id = []
        spk_id_format = []
        spk_id_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        df = pd.read_csv(os.path.join(data_folder, "data", split) + "_data.csv")
        for i in range(len(df)):
            ID.append(ID_start + i)
            signal = read_audio(os.path.join(data_folder, df.path[i]))
            duration.append(signal.shape[0] / 16000)

            wav.append(os.path.join(data_folder, df.path[i]))
            wav_format.append("wav")
            wav_opts.append(None)

            spk_id.append(df.speakerId[i])
            spk_id_format.append("string")
            spk_id_opts.append(None)

            transcript_ = df.transcription[i]
            transcript.append(transcript_)
            transcript_format.append("string")
            transcript_opts.append(None)

            semantics_ = (
                '{"action:" "'
                + df.action[i]
                + '"| "object": "'
                + df.object[i]
                + '"| "location": "'
                + df.location[i]
                + '"}'
            )
            semantics.append(semantics_)
            semantics_format.append("string")
            semantics_opts.append(None)

        new_df = pd.DataFrame(
            {
                "ID": ID,
                "duration": duration,
                "wav": wav,
                "spk_id": spk_id,
                "semantics": semantics,
                "transcript": transcript,
            }
        )
        new_df.to_csv(new_filename, index=False)
        ID_start += len(df)
