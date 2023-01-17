#!/bin/env python

from speechbrain.pretrained import EncoderASR, SepformerSeparation
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import torchaudio

import os
import re
import sys

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <custom_sepformer>")
    sys.exit(1)

print(sys.argv)
custom_model_path = sys.argv[1]

pid = os.getpid()
print("Process: ", pid)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model.from_pretrained("pyannote/segmentation", use_auth_token="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")
vad_model = VoiceActivityDetection(segmentation=model)
vad_model.instantiate(
    {
        # onset/offset activation thresholds
        "onset": 0.5, "offset": 0.5,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0
    }
)

sep_model = SepformerSeparation.from_hparams(
    source=custom_model_path,
    savedir='pretrained_models/sepformer-whamr16k',
    run_opts={"device": device},
)

asr_model = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-librispeech",
    savedir="pretrained_models/asr-wav2vec2-librispeech",
    run_opts={"device": device},
)

libri_css = Path("/workspace/data/LibriCSS/monaural/segments")
result_dir = Path(
    "/workspace/results/LibriCSS/asr-wav2vec2-librispeech_sepformer-{sep_model}_vad-pyannote/continuous_separation".format(
        sep_model=os.path.basename(custom_model_path)
    )
)

if os.path.exists(result_dir):
    print("{} exists! Refusing to overwrite it!".format(result_dir))
    sys.exit(1)

print("Saving results into: ", result_dir, file=sys.stderr)


def text2ctm(text, start_time, end_time, channel=0, conf=1.0):
    words = text.split()
    if len(words) == 0:
        return []

    dur = (end_time - 0.05 - start_time) / len(words)
    st = start_time
    ctms = []
    for wrd in words:
        ctms.append((channel, st, dur, wrd, conf))
        st += dur
    return ctms


def apply_pipeline(sep_model, vad_model, asr_model, wav_file, tmp_dir=".", tmp_name="tmp"):
    boundaries = vad_model(wav_file)
    mixture, sr = torchaudio.load(wav_file)

    ctm_results = []
    output_streams = []

    for segment in boundaries.get_timeline().support():
        sti, eti = int(segment.start * sr), int(segment.end * sr)
        segmented_mixture = mixture[:, range(sti, eti)]

        est_sources = sep_model(segmented_mixture)  # shape BxTxS
        output_streams.append((est_sources, sti, eti))

        for source_i, est_source in enumerate(est_sources.permute(2, 0, 1)):
            try:
                hyps, tokens = asr_model.transcribe_batch(est_source, torch.tensor([1.0]))
            except RuntimeError as e:
                print(e, file=sys.stderr)
                continue
            output = hyps[0]
            print(tmp_name, f"{segment.start:.2f}", f"{segment.end:.2f}", output, file=sys.stderr)
            ctm_results += text2ctm(output, segment.start, segment.end)

    recordings = process_streams(output_streams, mixture.shape[1])
    save_recordings(recordings, tmp_dir, tmp_name, sr)
    return ctm_results


def process_streams(output_streams, wav_len):
    first_ests, _, _ = output_streams[0]
    recordings = torch.zeros(first_ests.shape)
    for est_srcs, start_n, end_n in output_streams:
        recordings[:, start_n:end_n, :] = est_srcs
    return recordings


def save_recordings(recs, dirname, file_basename, sr=16000):
    for i, rec in enumerate(recs.permute(2, 0, 1)):
        torchaudio.save(os.path.join(dirname, f"{file_basename}_{i}.wav"), rec)


def main(args):
    ptrn = re.compile(".*(overlap_ratio[^/]+)")
    overlap_ratios = {}

    for wav_file in tqdm(glob(str(libri_css / "*/*.wav"))[::-1]):
        wavdir = os.path.dirname(wav_file)
        m = ptrn.match(os.path.abspath(wavdir))
        session = m.group(1)
        _, _, overlap_ratio, *tmp = session.split("_")

        overlap_ratios[overlap_ratio] = wav_file
        basename = os.path.splitext(os.path.basename(wav_file))[0]
        ctmfile = result_dir / session / (basename + ".ctm")
        if not ctmfile.parent.exists():
            ctmfile.parent.mkdir(parents=True)

        results = apply_pipeline(
            sep_model, vad_model, asr_model, wav_file, tmp_dir=result_dir / session, tmp_name=basename
        )

        with open(ctmfile, "w") as f:
            for channel, start_time, duration, word, confidence in results:
                f.write(
                    f"{session}_{basename} 0 {start_time} {duration} {word} {confidence}\n"
                )
    print("Success!")


main(None)
