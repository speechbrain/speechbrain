import os
import torchaudio

import logging

logger = logging.getLogger(__name__)

SAMPLERATE = 16000

def create_csv(save_folder, info_lst, split, select_n_sentences=None):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    info_lst : list
        The list of info (including id, wav, words and spk_id) from wav files of a given data split.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.
    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "wav", "wrd", "spk_id", "duration"]]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for info in info_lst:

        snt_id = info[0]
        spk_id = info[3]
        
        signal, fs = torchaudio.load(info[1])
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE

        csv_line = [
            snt_id,
            info[1],
            str(info[2]),
            spk_id,
            str(duration)
        ]

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1

        if select_n_sentences is not None and snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        ##up
        #print(csv_lines[4])
        #csv_lines[1:].sort(key=float(csv_lines[1:][4]))
        sorted(csv_lines[1:], key=lambda x:x[4])

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
