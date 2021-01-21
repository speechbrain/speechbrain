import os
import jsonlines

try:
    import pandas as pd
except ModuleNotFoundError:
    print("Error: pandas must be installed to run this recipe.")
    print("Install using `pip install pandas`.")
from speechbrain.dataio.dataio import read_wav_soundfile, merge_csvs

# from speechbrain.utils.data_utils import download_file


def prepare_SLURP(data_folder, slu_type, train_splits):
    """
    This function prepares the SLURP dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to SLURP dataset.
    slu_type : one of the following:

      "direct":{input=audio, output=semantics}
      "multistage":{input=audio, output=semantics} (using ASR transcripts in the middle)
      "decoupled":{input=transcript, output=semantics} (using ground-truth transcripts)

    train_splits : list of splits to be joined to form train .csv
    """

    # If the data folders do not exist, we need to extract the data
    # if not os.path.isdir(os.path.join(data_folder, "train-synth")):
    # Check for zip file and download if it doesn't exist
    #    zip_location = os.path.join(data_folder, "timers-and-such.zip")
    #    if not os.path.exists(zip_location):
    #        url = "https://zenodo.org/record/4110812/files/timers-and-such.zip?download=1"
    #        download_file(url, zip_location, unpack=True)
    #    else:
    #        print("Extracting timers-and-such.zip...")
    #        shutil.unpack_archive(zip_location, data_folder)

    splits = [
        "train",
        "train_synthetic",
        "devel",
        "test",
    ]
    id = 0
    for split in splits:
        new_filename = (
            os.path.join(data_folder, split) + "-type=%s.csv" % slu_type
        )
        if os.path.exists(new_filename):
            continue
        print("Preparing %s..." % new_filename)

        IDs = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        with jsonlines.open(
            os.path.join(data_folder, "dataset/slurp/" + split + ".jsonl")
        ) as reader:
            for obj in reader:
                scenario = obj["scenario"]
                action = obj["action"]
                sentence_annotation = obj["sentence_annotation"]
                num_entities = sentence_annotation.count("[")
                entities = []
                for slot in range(num_entities):
                    type = (
                        sentence_annotation.split("[")[slot + 1]
                        .split("]")[0]
                        .split(":")[0]
                        .strip()
                    )
                    filler = (
                        sentence_annotation.split("[")[slot + 1]
                        .split("]")[0]
                        .split(":")[1]
                        .strip()
                    )
                    entities.append({"type": type, "filler": filler})
                for recording in obj["recordings"]:
                    IDs.append(id)
                    if "synthetic" in split:
                        audio_folder = "scripts/audio/slurp_synth/"
                    else:
                        audio_folder = "scripts/audio/slurp_real/"
                    path = os.path.join(
                        data_folder, audio_folder, recording["file"]
                    )
                    signal = read_wav_soundfile(path)
                    duration.append(signal.shape[0] / 16000)

                    wav.append(path)
                    wav_format.append("flac")
                    wav_opts.append(None)

                    transcript_ = obj["sentence"]
                    if slu_type == "decoupled":
                        transcript_ = transcript_.upper()
                    transcript.append(transcript_)
                    transcript_format.append("string")
                    transcript_opts.append(None)

                    semantics_dict = {
                        "scenario": scenario,
                        "action": action,
                        "entities": entities,
                    }
                    semantics_ = str(semantics_dict).replace(
                        ",", "|"
                    )  # Commas in dict will make using csv files tricky; replace with pipe.
                    semantics.append(semantics_)
                    semantics_format.append("string")
                    semantics_opts.append(None)
                    id += 1

        df = pd.DataFrame(
            {
                "ID": IDs,
                "duration": duration,
                "wav": wav,
                "wav_format": wav_format,
                "wav_opts": wav_opts,
                "semantics": semantics,
                "semantics_format": semantics_format,
                "semantics_opts": semantics_opts,
                "transcript": transcript,
                "transcript_format": transcript_format,
                "transcript_opts": transcript_opts,
            }
        )
        df.to_csv(new_filename, index=False)

    # Merge train splits
    train_splits = [split + "-type=%s.csv" % slu_type for split in train_splits]
    merge_csvs(data_folder, train_splits, "train-type=%s.csv" % slu_type)
