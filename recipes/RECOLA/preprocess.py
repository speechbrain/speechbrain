import glob
import json
import os
import torchaudio
from pathlib import Path


def preprocess_RECOLA(
    data_path_audio,
    data_path_arousal,
    data_path_valence,
    experiment_folder,
    data_processed_folder,
):
    """
    Transforms audio files and makes the json files for all the data and different partitions

    Arguments
    ---------
    data_path_audio : str
        The path to the folder containing RECOLA audio .wav files
    data_path_arousal : str
        The path to arousal annotations of RECOLA (2016 format .arff files)
    data_path_valence : str
        The path to valence annotations of RECOLA (2016 format .arff files)
    experiment_folder : str
        The experiments folder, used to store json files for partitions and other information
    data_processed_folder : str
        The folder containing the processed data

    Example
    -------

    >>> preprocess_RECOLA("./RECOLA_2016/recordings_audio",
                          "./RECOLA_2016/ratings_gold_standard/arousal",
                          "./RECOLA_2016/ratings_gold_standard/valence",
                          "./Experiments",
                          "./Experiments/Audio")
    """

    # Avoiding reprocessing
    if os.path.exists(os.path.join(experiment_folder, "data.json")):
        print("RECOLA is already processed")
        return

    originalAudioPath = data_path_audio
    jsonFolderPath = experiment_folder
    arousalARFFolderPath = data_path_arousal
    valenceARFFolderPath = data_path_valence
    ProcessedAudioPath = data_processed_folder
    arousalCSVFolderPath = os.path.join(
        experiment_folder, "ratings_gold_standard", "arousal"
    )
    valenceCSVFolderPath = os.path.join(
        experiment_folder, "ratings_gold_standard", "valence"
    )

    writeWavFiles(originalAudioPath, ProcessedAudioPath)
    arffs2csvs(arousalARFFolderPath, arousalCSVFolderPath)
    arffs2csvs(valenceARFFolderPath, valenceCSVFolderPath)

    audioFilesPaths = glob.glob(os.path.join(ProcessedAudioPath, "*.wav"))

    allFilesInfo = {}
    trainFilesInfo = {}
    devFilesInfo = {}
    testFilesInfo = {}
    for i, filePath in enumerate(audioFilesPaths):
        utt_id, myDict = makeDict(
            filePath, arousalCSVFolderPath, valenceCSVFolderPath
        )
        allFilesInfo[utt_id] = myDict
        if "train_" in utt_id:
            trainFilesInfo[utt_id] = myDict
        if "dev_" in utt_id:
            devFilesInfo[utt_id] = myDict
        if "test_" in utt_id:
            testFilesInfo[utt_id] = myDict
        printProgressBar(
            i + 1,
            len(audioFilesPaths),
            prefix="Processing Files:",
            suffix="Complete",
        )
    with open(os.path.join(jsonFolderPath, "data.json"), "w") as fp:
        json.dump(allFilesInfo, fp, indent=4)
    with open(os.path.join(jsonFolderPath, "train.json"), "w") as fp:
        json.dump(trainFilesInfo, fp, indent=4)
    with open(os.path.join(jsonFolderPath, "dev.json"), "w") as fp:
        json.dump(devFilesInfo, fp, indent=4)
    with open(os.path.join(jsonFolderPath, "test.json"), "w") as fp:
        json.dump(testFilesInfo, fp, indent=4)


def makeDict(
    filePath: str, arousalCSVFolderPath: str, valenceCSVFolderPath: str
) -> (str, dict):
    """
    Gets the audio file path and produces a dictionary with necessary information
    """
    utt_id = Path(filePath).stem
    spk_id = utt_id
    duration = torchaudio.info(filePath).num_frames
    myDict = {
        "wav_path": filePath,
        "duration": duration,
        "spk_id": spk_id,
        "arousal_path": os.path.join(arousalCSVFolderPath, utt_id + ".csv"),
        "valence_path": os.path.join(valenceCSVFolderPath, utt_id + ".csv"),
    }
    return utt_id, myDict


def arffs2csvs(arffFolder: str, csvFolder: str) -> None:
    """
    Transform arff files in a folder for RECOLA dataset to csv ones
    """
    filesPath = os.path.join(arffFolder, "**", "*.arff")
    filesPaths = glob.glob(filesPath, recursive=True)
    for i, filePath in enumerate(filesPaths):
        fileName = Path(filePath).stem
        outPath = os.path.join(csvFolder, fileName + ".csv")
        if os.path.exists(outPath):
            continue
        directory = os.path.dirname(outPath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        arff2csv(filePath, csv_path=outPath)


def arff2csv(
    arff_path: str, csv_path: str = None, _encoding: str = "utf8"
) -> None:
    """
    Transform one arff file for RECOLA dataset to csv
    """
    with open(arff_path, "r", encoding=_encoding) as fr:
        attributes = []
        if csv_path is None:
            csv_path = arff_path[:-4] + "csv"  # *.arff -> *.csv
        write_sw = False
        with open(csv_path, "w", encoding=_encoding) as fw:
            for line in fr.readlines():
                if write_sw:
                    if line == "":
                        print("emp")
                    if line != "\n":
                        fw.write(line)
                elif "@data" in line:
                    fw.write(",".join(attributes) + "\n")
                    write_sw = True
                elif "@attribute" in line:
                    attributes.append(
                        line.split()[1]
                    )  # @attribute attribute_tag numeric
    print("Convert {} to {}.".format(arff_path, csv_path))


def writeWavFiles(wavsFolder: str, outFolder: str, allInOne=False) -> None:
    """
    Writes wav files in a specific format of 16 bit integer at 16k rate
    `allInOne` flag puts all the new audio files into the same folder, otherwise by default the original relative paths are preserved
    """
    wavFiles = glob.glob(os.path.join(wavsFolder, "*.wav"), recursive=True)
    for i, filePath in enumerate(wavFiles):
        print("filePath", filePath)
        printProgressBar(
            i + 1,
            len(wavFiles),
            prefix="Transforming audio files:",
            suffix="complete",
        )
        fileDirectory = os.path.split(filePath)[0]
        newName = os.path.split(filePath)[-1]
        newName = newName.split(".")[:-1] + [".wav"]
        newName = "".join(newName)
        fileNewPath = fileDirectory.replace(
            wavsFolder.replace("/**", ""), outFolder
        )
        fileNewPath = os.path.join(fileNewPath, newName)
        if allInOne:
            fileNewPath = os.path.join(outFolder, newName)
        directory = os.path.dirname(fileNewPath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(fileNewPath):
            continue
        os.system(
            "ffmpeg -i "
            + filePath
            + ' -ar 16000 -ac 1 -c:a pcm_s16le -af "volume=0dB" -hide_banner -v 0 -y '
            + fileNewPath
        )


def printProgressBar(
    iteration: int,
    total: int,
    prefix="",
    suffix="",
    decimals=1,
    length="fit",
    fill="█",
) -> None:
    """
    Prints a progress bar on the terminal
    """
    rows, columns = (
        os.popen("stty size", "r").read().split()
    )  # checks how wide the terminal width is
    if length == "fit":
        length = int(columns) // 2
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
    if iteration == total:  # go to new line when the progress bar is finished
        print()


# if __name__ == "__main__":
#     main()
