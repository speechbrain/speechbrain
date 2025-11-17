
"""
Myst data preparation (SpeechBrain-style) — silence detection removed.

This mirrors the API of `librispeech_prepare.py` while adding zero-shot
WER filtering across all splits (train/valid/test).

Outputs CSVs with columns:
    ID,duration,wav,spk_id,wrd

Expected layout per split directory:
    <data_folder>/<split>/**/<audio>.(wav|flac|mp3|m4a|ogg)
    with required sidecar transcripts: <audio>.trn

Authors: Thomas Rolland 2025
"""

import csv
import functools
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from tqdm import tqdm
import re


# Optional dependencies used by LibriSpeech prep
try:
    from speechbrain.dataio.dataio import (
        load_pkl,
        merge_csvs,
        read_audio_info,
        save_pkl,
    )
    from speechbrain.utils.data_utils import get_all_files
    from speechbrain.utils.logger import get_logger
    from speechbrain.utils.parallel import parallel_map
except Exception:  # graceful fallback if SpeechBrain is not present
    load_pkl = None
    merge_csvs = None
    read_audio_info = None
    save_pkl = None
    parallel_map = None
    def get_all_files(folder, match_and=None, match_or=None, exclude_or=None):
        out = []
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                path = os.path.join(root, fn)
                if match_or and not any(path.endswith(m) for m in match_or):
                    continue
                if match_and and not all(m in path for m in match_and):
                    continue
                if exclude_or and any(ex in path for ex in exclude_or):
                    continue
                out.append(path)
        return out
    class _Logger:
        def info(self, *a, **k): print(*a)
        def warning(self, *a, **k): print(*a)
    def get_logger(name): return _Logger()

logger = get_logger(__name__)
OPT_FILE = "opt_myst_prepare.pkl"

# Audio / text utilities
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
try:
    from whisper_normalizer.english import EnglishTextNormalizer  # type: ignore
except Exception:
    EnglishTextNormalizer = None

# ASR + WER (optional)
try:
    from faster_whisper import WhisperModel  # type: ignore
    _HAS_FASTER = True
except Exception:
    _HAS_FASTER = False
    WhisperModel = None  # type: ignore
try:
    import whisper  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
try:
    from jiwer import wer  # type: ignore
    _HAS_JIWER = True
except Exception:
    _HAS_JIWER = False

# librosa for duration (fallback if read_audio_info absent)
try:
    import librosa
except Exception:
    librosa = None


@dataclass
class MystRow:
    snt_id: str
    spk_id: str
    duration: float
    file_path: str
    words: str


def _normalize_text(text: str) -> str:
    """Normalize a transcript string for Whisper (also to remove some formating errors in transcriptions).

    Uses `whisper_normalizer.EnglishTextNormalizer` if available,
    otherwise just strips leading / trailing whitespace.

    Arguments
    ---------
    text : str
        Raw transcript text.

    Returns
    -------
    str
        Normalized text.
    """

    if not text:
        return text
    if EnglishTextNormalizer is None:
        return text.strip()
    return EnglishTextNormalizer()(text).strip()


def _sidecar_text(audio_path: str) -> Optional[str]:
    """Load the transcript for an audio file, if exists.

    Looks for a `.trn` file with the same basename as the audio and
    returns its contents if it exists and can be read.

    Arguments
    ---------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    str or None
        The transcript text, or None if it cannot be loaded.
    """

    p = Path(audio_path)
    txt = p.with_suffix(".trn")
    if txt.exists():
        try:
            return txt.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def _compute_duration(path: str) -> float:
    """Compute the duration (in seconds) of an audio file.

    Uses `speechbrain.dataio.dataio.read_audio_info` when available,
    falling back to `librosa` otherwise.

    Arguments
    ---------
    path : str
        Path to the audio file.

    Returns
    -------
    float
        Duration of the audio in seconds.
    """

    if read_audio_info is not None:
        info = read_audio_info(path)
        return float(info.num_frames) / float(info.sample_rate)
    if librosa is None:
        raise RuntimeError("Neither speechbrain.read_audio_info nor librosa is available to compute duration.")
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        y, sr = librosa.load(path, sr=16000, mono=True)
        return float(len(y) / sr)


def _derive_ids(audio_file: str, split_root: Path) -> Tuple[str, str]:
    """Derive sentence and speaker IDs from an audio path.

    The sentence ID is built from the relative path (without extension)
    and the speaker ID is derived from the parent directory name.

    Arguments
    ---------
    audio_file : str
        Path to the audio file.
    split_root : Path
        Root directory of the current split (train/valid/test).

    Returns
    -------
    tuple
        (snt_id, spk_id) pair for the utterance.
    """
    rel = Path(audio_file).resolve().relative_to(split_root.resolve())
    snt_id = "_".join(rel.with_suffix("").parts)
    spk_id = Path(audio_file).parent.name
    return snt_id, spk_id


def _find_audio(split_dir: Path) -> List[str]:
    """Find all audio files for a given split.

    Recursively scans the split directory for files with supported
    audio extensions.

    Arguments
    ---------
    split_dir : Path
        Root directory of the split.

    Returns
    -------
    list
        Sorted list of audio file paths.
    """
    wavs = get_all_files(str(split_dir), match_or=AUDIO_EXTS)
    wavs.sort()
    return wavs


def _process_line(audio_file: str, split_root: Path) -> Optional[MystRow]:
    """Create a `MystRow` from a single audio file.

    Loads thetranscript, filters out unsuitable utterances
    (no text, special tags, very short or very long), normalizes the
    text, computes duration, and packages everything into a `MystRow`.

    Arguments
    ---------
    audio_file : str
        Path to the audio file.
    split_root : Path
        Root directory of the current split (train/valid/test).

    Returns
    -------
    MystRow or None
        A populated `MystRow` if the utterance passes all filters,
        otherwise None.
    """

    txt = _sidecar_text(audio_file)
    if txt is None:
        # Text is mandatory for all utterance (select only supervised data)
        return None

    # Skip files with no speech
    if txt in ["<DISCARD>", "<NO SIGNAL>", "<SILENCE>"]:
        return None

    # Skip one- and two-word files
    if len(txt.split()) <= 2:
        return None

    snt_id, spk_id = _derive_ids(audio_file, split_root)
    wrds = _normalize_text(txt)
    if "$" in wrds: # We have one utterance with $ that will raises errors later if we don't remove it now
        wrds =  re.sub(r'\$(\d+)', r'\1 dollars', wrds)
    duration = _compute_duration(audio_file)

    # Skip files longer than 30s
    if duration > 30.0:
        return None

    return MystRow(
        snt_id=snt_id,
        spk_id=spk_id,
        duration=duration,
        file_path=audio_file,
        words=wrds,
    )


class _ASRBackend:
    """Thin wrapper around Faster-Whisper or OpenAI Whisper.

    Selects the available backend at construction time and exposes
    a single `transcribe(path)` method used for WER filtering.
    """
    def __init__(self, model_name: str = "small", device: Optional[str] = None):
        if _HAS_FASTER:
            self.kind = "faster"
            self.model = WhisperModel(model_name, device=device or "auto", compute_type="float32")
        elif _HAS_OPENAI:
            self.kind = "openai"
            self.model = whisper.load_model(model_name, device=device or None)  # type: ignore
        else:
            raise RuntimeError("No ASR backend. Install faster-whisper or openai-whisper.")

    def transcribe(self, path: str) -> str:
        """Transcribe an audio file to text.

        Arguments
        ---------
        path : str
            Path to the audio file.

        Returns
        -------
        str
            Normalized ASR hypothesis.
        """
        if self.kind == "faster":
            segs, _ = self.model.transcribe(path, task="transcribe", language="en")
            hyp = " ".join(s.text for s in segs).strip()
        else:
            out = self.model.transcribe(path, task="transcribe", language="en")
            hyp = out.get("text", "").strip()
        return _normalize_text(hyp)


def _apply_wer_filter(rows: List[MystRow], threshold: float, asr_model: str, device: Optional[str]) -> List[MystRow]:
    """Filter rows using zero-shot ASR and WER.

    Runs an ASR model (whisper) on each utterance and keeps only those whose
    word error rate (WER) is below or equal to the given threshold. This is done
    to remove transcription errors present in the dataset.

    Arguments
    ---------
    rows : list of MystRow
        Input rows to filter.
    threshold : float
        Maximum allowed WER (e.g., 0.3 for 30%).
    asr_model : str
        Whisper model size/name to use for ASR.
    device : str, optional
        Device for ASR inference (e.g., 'cpu', 'cuda').

    Returns
    -------
    list of MystRow
        The subset of rows that pass the WER filter.
    """
    if threshold is None:
        return rows
    if not _HAS_JIWER:
        raise RuntimeError("jiwer is required for WER filtering. pip install jiwer")
    asr = _ASRBackend(asr_model, device=device)
    kept: List[MystRow] = []
    for r in tqdm(rows, desc="WER filtering"):
        hyp = asr.transcribe(r.file_path)
        ref = _normalize_text(r.words)
        hyp = _normalize_text(hyp)
        score = wer(ref, hyp)
        if score <= threshold:
            kept.append(r)
    return kept


def _write_csv(rows: List[MystRow], csv_file: str) -> None:
    """Write a list of rows to a CSV file.

    The CSV has columns: ID, duration, wav, spk_id, wrd.

    Arguments
    ---------
    rows : list of MystRow
        Rows to serialize.
    csv_file : str
        Path to the output CSV file.

    Returns
    -------
    None
    """
    if not rows:
        logger.warning(f"No rows to write for {csv_file}")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    lines = [["ID","duration","wav","spk_id","wrd"]]
    for r in rows:
        lines.append([r.snt_id, r.duration, r.file_path, r.spk_id, r.words])
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines)


def _maybe_skip(save_folder: str, splits: Sequence[str], conf: dict) -> bool:
    """Decide whether data preparation can be skipped.

    Checks if all requested split CSVs already exist and, when possible,
    compares the stored options pickle with the current configuration.

    Arguments
    ---------
    save_folder : str
        Directory where CSVs and the options pickle are stored.
    splits : sequence of str
        Names of the splits that must be present.
    conf : dict
        Current configuration options.

    Returns
    -------
    bool
        True if preparation can be skipped safely, False otherwise.
    """

    # Do we already have all desired CSVs?
    skip = True
    for s in splits:
        if not os.path.isfile(os.path.join(save_folder, s + ".csv")):
            skip = False
    # Option equality check (if speechbrain load/save pkl available)
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip and load_pkl is not None and os.path.isfile(save_opt):
        try:
            old = load_pkl(save_opt)
            if old == conf:
                return True
        except Exception:
            pass
        return False
    return skip


def prepare_myst(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    enable_wer_filter=True,
    wer_threshold=0.3,
    asr_model="small",
    device="cuda",
    skip_prep=False,
):
    """Prepare Myst-style CSVs (train/valid/test) with optional WER filtering.

    Arguments
    ---------
    data_folder : str
        Root folder containing split subfolders (e.g., train, valid, test).
    save_folder : str
        Where to store the output CSVs and options pickle.
    tr_splits, dev_splits, te_splits : list[str]
        Split names to process (subfolders inside data_folder).
    select_n_sentences : int, optional
        If set, limit the number of sentences per split.
    merge_lst : list, optional
        List of CSVs to merge after preparation.
    merge_name : str, optional
        Name for the merged CSV (written in save_folder).
    enable_wer_filter : bool
        If True, apply zero-shot WER filtering to all splits.
    wer_threshold : float
        Drop rows with WER > threshold (e.g., 0.3 = 30%).
    asr_model : str
        Whisper model size/name.
    device : str, optional
        Backend device (e.g., 'cpu', 'cuda').
    skip_prep : bool
        If True, skip if files already prepared with same options.
    """

    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "tr_splits": tr_splits,
        "dev_splits": dev_splits,
        "te_splits": te_splits,
        "select_n_sentences": select_n_sentences,
        "merge_lst": merge_lst,
        "merge_name": merge_name,
        "enable_wer_filter": enable_wer_filter,
        "wer_threshold": wer_threshold,
        "asr_model": asr_model,
        "device": device,
    }

    os.makedirs(save_folder, exist_ok=True)

    splits = []
    splits += [(s, os.path.join(data_folder, s)) for s in tr_splits]
    splits += [(s, os.path.join(data_folder, s)) for s in dev_splits]
    splits += [(s, os.path.join(data_folder, s)) for s in te_splits]

    if skip_prep and _maybe_skip(save_folder, [s for s,_ in splits], conf):
        logger.info("Preparation already completed with same options. Skipping.")
        return

    # Process each split
    for split_name, split_path in splits:
        csv_file = os.path.join(save_folder, split_name + ".csv")
        if os.path.exists(csv_file):
            logger.info(f"Csv file {csv_file} already exists, not recreating.")
            continue

        logger.info(f"Creating csv list for split '{split_name}' at {csv_file} ...")
        split_root = Path(split_path)
        if not split_root.exists():
            logger.warning(f"Split folder not found: {split_root}. Skipping {split_name}.")
            continue

        wav_lst = _find_audio(split_root)
        logger.info(f"Found {len(wav_lst)} audio files for split '{split_name}'")

        # Map to rows (parallel if available)
        if parallel_map is not None:
            logger.info(f"Parallel mapping")

            line_proc = functools.partial(_process_line, split_root=split_root)
            rows = list(filter(None, parallel_map(line_proc, wav_lst)))
        else:
            logger.info(f"Sequential mapping")

            rows = []
            for w in wav_lst:
                r = _process_line(w, split_root=split_root)
                if r is not None:
                    rows.append(r)
        logger.info(f"Before WER filtering {len(rows)} audio files for split '{split_name}'")

        # Optional selection
        if select_n_sentences is not None:
            rows = rows[:select_n_sentences]

        # Optional WER filter
        if enable_wer_filter and wer_threshold is not None:
            rows = _apply_wer_filter(rows, wer_threshold, asr_model, device)
        logger.info(f"After WER filtering {len(rows)} audio files for split '{split_name}'")

        # Deterministic order
        rows.sort(key=lambda r: (r.spk_id, r.snt_id))

        _write_csv(rows, csv_file)
        logger.info(f"[{split_name}] {len(rows)} rows -> {csv_file}")

    # Save options (if speechbrain present)
    if save_pkl is not None:
        try:
            save_pkl(conf, os.path.join(save_folder, OPT_FILE))
        except Exception:
            pass

    # Optional merge (could be useful if use of unsupervised/semi supersived data)
    if merge_lst and merge_name:
        if merge_csvs is None:
            logger.warning("merge_csvs not available (speechbrain not installed). Skipping merge.")
        else:
            csv_to_merge = [os.path.join(save_folder, s + ".csv") for s in merge_lst]
            merged_csv = os.path.join(save_folder, merge_name + ".csv")
            merge_csvs(csv_to_merge, merged_csv)
            logger.info(f"Merged -> {merged_csv}")
