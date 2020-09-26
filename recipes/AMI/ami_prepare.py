"""
Data preparation.

Download: http://groups.inf.ed.ac.uk/ami/download/

Prepares csv from manual annotations "segments/" using RTTM format (Oracle VAD).
"""

import os
import sys  # noqa F401
import logging
import xml.etree.ElementTree as et
import glob
import csv
from ami_splits import get_AMI_split

from speechbrain.data_io.data_io import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_ami_prepare.pkl"
TRAIN_CSV = "ami_train.subsegments.csv"
DEV_CSV = "ami_dev.subsegments.csv"
EVAL_CSV = "ami_eval.subsegments.csv"
SAMPLERATE = 16000


def get_RTTM_per_rec(segs, spkrs_list, rec_id):
    """Prepares rttm for each recording
    """

    rttm = []

    # Prepare header
    for spkr_id in spkrs_list:
        # e.g. SPKR-INFO ES2008c 0 <NA> <NA> <NA> unknown ES2008c.A_PM <NA> <NA>
        line = (
            "SPKR-INFO "
            + rec_id
            + " 0 <NA> <NA> <NA> unknown "
            + spkr_id
            + " <NA> <NA>"
        )
        rttm.append(line)

    # Append remaining lines
    for row in segs:
        # e.g. SPEAKER ES2008c 0 37.880 0.590 <NA> <NA> ES2008c.A_PM <NA> <NA>

        if float(row[1]) < float(row[0]):
            msg1 = (
                "Incorrect Annotation Found: transcriber_start (%s) > transcriber_start (%s)"
                % (row[0], row[1])
            )
            msg2 = "Excluding incorrect row from RTTM : %s, %s, %s, %s" % (
                rec_id,
                row[0],
                str(round(float(row[1]) - float(row[0]), 4)),
                str(row[2]),
            )
            logger.info(msg1)
            logger.info(msg2)
            continue

        line = (
            "SPEAKER "
            + rec_id
            + " 0 "
            + str(round(float(row[0]), 4))
            + " "
            + str(round(float(row[1]) - float(row[0]), 4))
            + " <NA> <NA> "
            + str(row[2])
            + " <NA> <NA>"
        )
        rttm.append(line)

    return rttm


def prepare_segs_for_RTTM(
    list_ids, out_rttm_file, audio_dir, annot_dir, split_type
):

    RTTM = []  # All RTTMs clubbed together for a given dataset
    for main_meet_id in list_ids:

        # Skip all TNO (later just skip it from dev and test)
        # SPEAKER DIARISATION USING 2D SELF-ATTENTIVE COMBINATION OF EMBEDDINGS, ICASSP 2019
        if main_meet_id.startswith("TS") and split_type != "train":
            msg = (
                "Skipping TNO meeting in AMI "
                + str(split_type)
                + " set : "
                + str(main_meet_id)
            )
            logger.info(msg)
            continue

        list_sessions = glob.glob(audio_dir + "/" + main_meet_id + "*")
        list_sessions.sort()

        for sess in list_sessions:
            rec_id = os.path.basename(sess)
            path = annot_dir + "/segments/" + rec_id
            f = path + ".*.segments.xml"
            list_spkr_xmls = glob.glob(f)
            list_spkr_xmls.sort()  # A, B, C, D, E (Speakers)
            segs = []
            spkrs_list = []  # Since non-scene recordings have 3-5 speakers

            for spkr_xml_file in list_spkr_xmls:

                # Speaker ID
                spkr = os.path.basename(spkr_xml_file).split(".")[1]
                spkr_ID = rec_id + "." + spkr
                spkrs_list.append(spkr_ID)  # used while get_RTTM_per_rec

                # Parse xml tree
                tree = et.parse(spkr_xml_file)
                root = tree.getroot()

                # Start, end and speaker_ID from xml file
                segs = segs + [
                    [
                        elem.attrib["transcriber_start"],
                        elem.attrib["transcriber_end"],
                        spkr_ID,
                    ]
                    for elem in root.iter("segment")
                ]

            # Sort rows as per start time (per recording)
            segs.sort(key=lambda x: float(x[0]))

            rttm_per_rec = get_RTTM_per_rec(segs, spkrs_list, rec_id)
            RTTM = RTTM + rttm_per_rec

    # Write 1 RTTM as groundtruth "fullref_eval.rttm"
    with open(out_rttm_file, "w") as f:
        for item in RTTM:
            f.write("%s\n" % item)


def is_overlapped(end1, start2):
    """Returns True if 2 segments overlap
    """
    if start2 > end1:
        return False
    else:
        return True


def merge_rttm_intervals(rttm_segs):
    """Merges segments if overlaped
    """
    # For one recording
    # rec_id = rttm_segs[0][1]
    rttm_segs.sort(key=lambda x: float(x[3]))

    # first_seg = rttm_segs[0] # first interval.. as it is
    merged_segs = [rttm_segs[0]]
    strt = float(rttm_segs[0][3])
    end = float(rttm_segs[0][3]) + float(rttm_segs[0][4])

    for row in rttm_segs[1:]:
        s = float(row[3])
        e = float(row[3]) + float(row[4])

        if is_overlapped(end, s):
            # Update end. strt will be same as in last segment
            # Just update last row in the merged_segs
            end = max(end, e)
            merged_segs[-1][3] = str(round(strt, 4))
            merged_segs[-1][4] = str(round((end - strt), 4))
            merged_segs[-1][7] = "overlap"  # previous_row[7] + '-'+ row[7]
        else:
            # Add a new disjoint segment
            strt = s
            end = e
            merged_segs.append(row)  # this will have 1 spkr ID

    return merged_segs


def get_subsegments(merged_segs, max_subseg_dur=3.0, overlap=1.5):
    """Divides the bigger segments into smaller sub-segments
    """
    shift = max_subseg_dur - overlap
    subsegments = []
    # These rows are in RTTM format
    for row in merged_segs:
        seg_dur = float(row[4])
        rec_id = row[1]
        if seg_dur > max_subseg_dur:
            num_subsegs = int(seg_dur / shift)
            # Usually, frame shift is 10ms
            # So assuming 0.01 sec as discrete step
            seg_start = float(row[3])
            seg_end = seg_start + seg_dur
            # Now divide this segment (new_row) in smaller subsegments
            for i in range(num_subsegs):
                subseg_start = seg_start + i * shift
                subseg_end = min(subseg_start + max_subseg_dur - 0.01, seg_end)
                subseg_dur = subseg_end - subseg_start

                new_row = [
                    "SPEAKER",
                    rec_id,
                    "0",
                    str(round(float(subseg_start), 4)),
                    str(round(float(subseg_dur), 4)),
                    "<NA>",
                    "<NA>",
                    row[7],
                    "<NA>",
                    "<NA>",
                ]

                subsegments.append(new_row)

                # Break if exceeding the boundary
                if subseg_end >= seg_end:
                    break
        else:
            subsegments.append(row)

    return subsegments


def prepare_csv(
    rttm_file, save_dir, data_dir, filename, max_subseg_dur, overlap, mic_type
):
    # Read RTTM, get unique meeting_IDs (from RTTM headers)
    # For each MeetingID.. select only that meetID -> merge -> subsegment -> csv -> append

    # Read RTTM
    RTTM = []
    with open(rttm_file, "r") as f:
        for line in f:
            entry = line[:-1]
            RTTM.append(entry)

    spkr_info = filter(lambda x: x.startswith("SPKR-INFO"), RTTM)
    rec_ids = list(set([row.split(" ")[1] for row in spkr_info]))
    rec_ids.sort()  # sorting just to make CSV look in proper sequence

    # For each recording merge segments and then perform subsegmentation
    MERGED_SEGMENTS = []
    SUBSEGMENTS = []
    for rec_id in rec_ids:
        segs_iter = filter(
            lambda x: x.startswith("SPEAKER " + str(rec_id)), RTTM
        )
        gt_rttm_segs = [row.split(" ") for row in segs_iter]

        # Merge, subsegment and convert to csv format.
        merged_segs = merge_rttm_intervals(
            gt_rttm_segs
        )  # We lose speaker_ID after merging
        MERGED_SEGMENTS = MERGED_SEGMENTS + merged_segs

        # Divide segments into smaller sub-segments
        subsegs = get_subsegments(merged_segs, max_subseg_dur, overlap)
        SUBSEGMENTS = SUBSEGMENTS + subsegs

    # Write segment AND sub-segments (in RTTM format)
    # Useful for verifing correctness of sub-segmentor module. Verify using md-eval.pl and check Missed Speech, FA Speech
    segs_file = save_dir + "/" + filename + ".segments.rttm"
    subsegment_file = save_dir + "/" + filename + ".subsegments.rttm"

    with open(segs_file, "w") as f:
        for row in MERGED_SEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    with open(subsegment_file, "w") as f:
        for row in SUBSEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    # Create CSV from subsegments
    csv_output_head = [
        ["ID", "duration", "wav", "wav_format", "wav_opts"]
    ]  # noqa E231

    entry = []
    for row in SUBSEGMENTS:
        rec_id = row[1]
        strt = str(round(float(row[3]), 4))
        end = str(round((float(row[3]) + float(row[4])), 4))

        subsegment_ID = rec_id + "_" + strt + "_" + end
        dur = row[4]
        wav_file_path = (
            data_dir
            + "/"
            + rec_id
            + "/audio/"
            + rec_id
            + ".Mix-"
            + mic_type
            + ".wav"
        )

        start_sample = int(float(strt) * SAMPLERATE)
        end_sample = int(float(end) * SAMPLERATE)
        # print ("HEY, " , start_sample, end_sample)

        start_stop = "start:" + str(start_sample) + " stop:" + str(end_sample)

        # start_stop = "start:" + strt + " stop:" + end

        # Composition of the csv_line
        csv_line = [
            subsegment_ID,
            dur,
            wav_file_path,
            "wav",
            start_stop,
        ]

        entry.append(csv_line)

    csv_output = csv_output_head + entry

    # Write csv file only for subsegments
    csv_file = save_dir + "/" + filename + ".subsegments.csv"
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    msg = "%s csv prepared" % (csv_file)
    logger.debug(msg)


def prepare_ami(
    data_folder,
    manual_annot_folder,
    save_folder,
    split_type="full_corpus_asr",
    mic_type="Lapel",
    vad_type="oracle",
    max_subseg_dur=3.0,
    overlap=1.5,
):
    """
    Prepares the csv files for the AMI dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    """

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "split_type": split_type,
        "mic_type": mic_type,
        "vad": vad_type,
        "max_subseg_dur": max_subseg_dur,
        "overlap": overlap,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput opt files
    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    splits = ["train", "dev", "eval"]
    if skip(splits, save_folder, conf):
        logger.info(
            "Skipping data preparation, as it was completed in previous run."
        )
        return

    msg = "\tCreating csv file for the VoxCeleb1 Dataset.."
    logger.debug(msg)

    # Get the split
    train_set, dev_set, eval_set = get_AMI_split(split_type)

    # Prepare RTTM from XML(manual annot) and store are groundtruth
    # Create ref_RTTM directory
    ref_dir = save_folder + "/ref_rttms/"
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)

    # Create reference RTTM files
    for i in splits:
        rttm_file = ref_dir + "/fullref_ami_" + i + ".rttm"
        if i == "train":
            prepare_segs_for_RTTM(
                train_set, rttm_file, data_folder, manual_annot_folder, i
            )
        if i == "dev":
            prepare_segs_for_RTTM(
                dev_set, rttm_file, data_folder, manual_annot_folder, i
            )
        if i == "eval":
            prepare_segs_for_RTTM(
                eval_set, rttm_file, data_folder, manual_annot_folder, i
            )

    # Create csv_files for splits
    csv_folder = os.path.join(save_folder, "csv")
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    for i in splits:
        rttm_file = ref_dir + "/fullref_ami_" + i + ".rttm"
        csv_filename_prefix = "ami_" + i
        prepare_csv(
            rttm_file,
            csv_folder,
            data_folder,
            csv_filename_prefix,
            max_subseg_dur,
            overlap,
            mic_type,
        )

    # Perform subsegmentation on large segments
    # Inp: largesegments, max_subseg_size, overlap (1sec)
    # train_subsegs = subsegmentor (train_segs, max_subseg_size, overlap)
    # dev_subsegs = subsegmentor (dev_segs, max_subseg_size, overlap)
    # eval_subsegs = subsegmentor (eval_segs, max_subseg_size, overlap)

    # Creating csv file using above subsegments
    # prepare_csv(train_set, dev_set, eval_set)

    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the timit data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "eval": EVAL_CSV,
    }
    for split in splits:
        # if not os.path.isfile(os.path.join(save_folder, split_files[split])):
        if not os.path.isfile(
            os.path.join(save_folder, "csv", split_files[split])
        ):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)  # noqa F821
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip
