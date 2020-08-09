import re
import subprocess
import numpy as np

FILE_IDS = re.compile(r"(?<=Speaker Diarization for).+(?=\*\*\*)")
SCORED_SPEAKER_TIME = re.compile(r"(?<=SCORED SPEAKER TIME =)[\d.]+")
MISS_SPEAKER_TIME = re.compile(r"(?<=MISSED SPEAKER TIME =)[\d.]+")
FA_SPEAKER_TIME = re.compile(r"(?<=FALARM SPEAKER TIME =)[\d.]+")
ERROR_SPEAKER_TIME = re.compile(r"(?<=SPEAKER ERROR TIME =)[\d.]+")


def rectify(arr):
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 1
    arr *= 100.0

    return arr


def DER(
    ref_rttm,
    sys_rttm,
    ignore_overlap=False,
    collar=0.25,
    individual_file_scores=False,
):
    mdEval = (
        "/home/mila/d/dawalatn/nauman/sb/speechbrain/tools/der_eval/md-eval.pl"
    )

    cmd = [
        mdEval,
        "-af",
        "-r",
        ref_rttm,
        "-s",
        sys_rttm,
        "-c",
        str(collar),
    ]
    if ignore_overlap:
        cmd.append("-1")

    # Parse md-eval output to extract by-file and total scores.
    stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    stdout = stdout.decode("utf-8")

    # Get all recording IDs
    file_ids = [m.strip() for m in FILE_IDS.findall(stdout)]
    file_ids = [
        file_id[2:] if file_id.startswith("f=") else file_id
        for file_id in file_ids
    ]

    scored_speaker_times = np.array(
        [float(m) for m in SCORED_SPEAKER_TIME.findall(stdout)]
    )

    miss_speaker_times = np.array(
        [float(m) for m in MISS_SPEAKER_TIME.findall(stdout)]
    )

    fa_speaker_times = np.array(
        [float(m) for m in FA_SPEAKER_TIME.findall(stdout)]
    )

    error_speaker_times = np.array(
        [float(m) for m in ERROR_SPEAKER_TIME.findall(stdout)]
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        tot_error_times = (
            miss_speaker_times + fa_speaker_times + error_speaker_times
        )
        miss_speaker_frac = miss_speaker_times / scored_speaker_times
        fa_speaker_frac = fa_speaker_times / scored_speaker_times
        sers_frac = error_speaker_times / scored_speaker_times
        ders_frac = tot_error_times / scored_speaker_times

    # Values in percentage of scored_speaker_time
    miss_speaker = rectify(miss_speaker_frac)
    fa_speaker = rectify(fa_speaker_frac)
    sers = rectify(sers_frac)
    ders = rectify(ders_frac)

    print(miss_speaker)
    print(fa_speaker)
    print(sers)
    print(ders)

    if individual_file_scores:
        return miss_speaker, fa_speaker, sers, ders
    else:
        return miss_speaker[-1], fa_speaker[-1], sers[-1], ders[-1]


if __name__ == "__main__":
    print("hello")
    ref_rttm = "/home/mila/d/dawalatn/nauman/sb/speechbrain/samples/rttm_samples/ref_rttm/IS1000a.rttm"
    sys_rttm = "/home/mila/d/dawalatn/nauman/sb/speechbrain/samples/rttm_samples/sys_rttm/IS1000a.rttm"

    miss_speaker, fa_speaker, sers, ders = DER(
        ref_rttm, sys_rttm, individual_files=True
    )
    print("MAIN")
    print(miss_speaker)
    print(fa_speaker)
    print(sers)
    print(ders)
