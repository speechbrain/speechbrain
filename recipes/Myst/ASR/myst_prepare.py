
"""
Myst data preparation (SpeechBrain-style).

This mirrors the API of `librispeech_prepare.py` while optionally applying
zero-shot WER filtering across all splits (train/valid/test). When enabled,
this step removes utterances whose transcript is too far from a Whisper decode.
This step is use to remove wrong transcription from the dataset.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

TextNormalizer = Callable[[str], str]
from tqdm import tqdm
import re


from speechbrain.dataio.dataio import (
    load_pkl,
    merge_csvs,
    read_audio_info,
    save_pkl,
)
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

from speechbrain.utils.metric_stats import ErrorRateStats
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def sb_wer(ref_tokens, hyp_tokens): # Quick WER computation using SpeechBrain's ErrorRateStats
    stats = ErrorRateStats()
    stats.append([0], [ref_tokens], [hyp_tokens])
    summary = stats.summarize()
    # Prefer standard WER key, fallback to alt naming if present
    if "WER" in summary:
        return float(summary["WER"])
    if "WER-w/o-trailing-silence" in summary:
        return float(summary["WER-w/o-trailing-silence"])
    return 100.0

logger = get_logger(__name__)
OPT_FILE = "opt_myst_prepare.pkl"

# Audio / text utilities
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


@dataclass
class MystRow:
    snt_id: str
    spk_id: str
    duration: float
    file_path: str
    words: str


def _normalize_text(
    text: str,
    normalizer: Optional[TextNormalizer] = None,
) -> str:
    """Normalize a transcript string for Whisper (also to remove some formating errors in transcriptions).

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
    if normalizer is not None and callable(normalizer):
        try:
            return normalizer(text).strip()
        except Exception:
            pass
    return text.strip()


def _get_normalizer_from_tokenizer(tokenizer) -> Optional[TextNormalizer]:
    """Extract a callable normalizer from a Whisper tokenizer if available."""
    for attr in ("normalize", "_normalize"):
        norm = getattr(tokenizer, attr, None)
        if norm is None:
            continue
        if callable(norm):
            return norm
        if hasattr(norm, "normalize"):
            return lambda text: norm.normalize(text)
    return None


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

    Uses `speechbrain.dataio.dataio.read_audio_info`.

    Arguments
    ---------
    path : str
        Path to the audio file.

    Returns
    -------
    float
        Duration of the audio in seconds.
    """

    info = read_audio_info(path)
    return float(info.num_frames) / float(info.sample_rate)


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


def _process_line(
    audio_file: str,
    split_root: Path,
    normalizer: Optional[TextNormalizer] = None,
) -> Optional[MystRow]:
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
    wrds = _normalize_text(txt, normalizer=normalizer)
    if "$" in wrds:
        # Replace the lone dollar pattern currently present in the corpus.
        wrds = re.sub(r"\$(\d+)", r"\1 dollars", wrds)
    duration = _compute_duration(audio_file)

    # Skip files longer than 30s or shorter than a second
    if duration > 30.0 or duration < 1.0:
        return None

    return MystRow(
        snt_id=snt_id,
        spk_id=spk_id,
        duration=duration,
        file_path=audio_file,
        words=wrds,
    )


class _ASRBackend:
    """Thin wrapper around Hugging Face Whisper ASR."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: Optional[str] = None,
        normalizer: Optional[TextNormalizer] = None,
        preloaded_asr: Optional[object] = None,
    ):
        self.device_index = 0 if device and device.startswith("cuda") else -1
        self.processor = None
        self.pipe = preloaded_asr
        if self.pipe is None:
            self.processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device_index,
            )
        tokenizer = getattr(self.processor, "tokenizer", None) or getattr(self.pipe, "tokenizer", None)
        self.normalizer = normalizer or _get_normalizer_from_tokenizer(tokenizer)

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
        out = self.pipe(path)
        hyp = out.get("text", "") if isinstance(out, dict) else str(out)
        return _normalize_text(hyp, normalizer=self.normalizer)


def _apply_wer_filter(
    rows: List[MystRow],
    threshold: float,
    asr_model: str,
    device: Optional[str],
    normalizer: Optional[TextNormalizer],
    asr_instance: Optional[object] = None,
) -> List[MystRow]:
    """Filter rows using zero-shot ASR and WER.

    Runs an ASR model (whisper) on each utterance and keeps only those whose
    word error rate (WER) is below or equal to the given threshold. This is done
    to remove transcription errors present in the dataset.

    Arguments
    ---------
    rows : list of MystRow
        Input rows to filter.
    threshold : float
        Maximum allowed WER (e.g., 30%).
    asr_model : str
        Whisper model size/name to use for ASR.
    device : str, optional
        Device for ASR inference (e.g., 'cpu', 'cuda').
    normalizer : callable, optional
        Normalizer (string -> string) to apply to both reference and hypothesis.
    asr_instance : HF pipeline, optional
        Preloaded Hugging Face ASR pipeline to reuse.

    Returns
    -------
    list of MystRow
        The subset of rows that pass the WER filter.
    """
    if threshold is None:
        return rows
    asr = _ASRBackend(
        asr_model,
        device=device,
        normalizer=normalizer,
        preloaded_asr=asr_instance,
    )
    norm_fn = asr.normalizer or normalizer
    kept: List[MystRow] = []
    for r in tqdm(rows, desc="WER filtering"):
        hyp = asr.transcribe(r.file_path)
        ref = _normalize_text(r.words, normalizer=norm_fn)
        hyp = _normalize_text(hyp, normalizer=norm_fn)
        score = sb_wer(ref.split(), hyp.split())
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
    if skip and os.path.isfile(save_opt):
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
    wer_threshold=30.0,
    asr_model="openai/whisper-large-v3",
    device="cuda",
    normalizer: Optional[TextNormalizer] = None,
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
        Drop rows with WER > threshold (e.g.,  30%).
    asr_model : str
        Whisper model size/name.
    device : str, optional
        Backend device (e.g., 'cpu', 'cuda').
    normalizer : callable, optional
        Text normalizer (string -> string). If None, a normalizer is pulled
        from the Whisper tokenizer (loading the model if needed).
    skip_prep : bool
        If True, skip if files already prepared with same options.
    """

    normalizer_label = normalizer.__class__.__name__ if normalizer is not None else "auto"

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
        "normalizer": normalizer_label,
    }

    os.makedirs(save_folder, exist_ok=True)

    splits = []
    splits += [(s, os.path.join(data_folder, s)) for s in tr_splits]
    splits += [(s, os.path.join(data_folder, s)) for s in dev_splits]
    splits += [(s, os.path.join(data_folder, s)) for s in te_splits]

    if skip_prep and _maybe_skip(save_folder, [s for s,_ in splits], conf):
        logger.info("Preparation already completed with same options. Skipping.")
        return

    resolved_normalizer = normalizer
    if resolved_normalizer is None:
        try:
            processor = AutoProcessor.from_pretrained(asr_model)
            resolved_normalizer = _get_normalizer_from_tokenizer(processor.tokenizer)
        except Exception as exc:
            logger.warning(f"Unable to fetch Whisper normalizer automatically: {exc}")

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

            line_proc = functools.partial(
                _process_line,
                split_root=split_root,
                normalizer=resolved_normalizer,
            )
            rows = list(filter(None, parallel_map(line_proc, wav_lst)))
        else:
            logger.info(f"Sequential mapping")

            rows = []
            for w in wav_lst:
                r = _process_line(
                    w,
                    split_root=split_root,
                    normalizer=resolved_normalizer,
                )
                if r is not None:
                    rows.append(r)
        logger.info(f"Before WER filtering {len(rows)} audio files for split '{split_name}'")

        # Optional selection
        if select_n_sentences is not None:
            rows = rows[:select_n_sentences]

        # Optional WER filter
        if enable_wer_filter and wer_threshold is not None:
            rows = _apply_wer_filter(
                rows,
                wer_threshold,
                asr_model,
                device,
                normalizer=resolved_normalizer,
            )
            logger.info(
                f"After WER filtering {len(rows)} audio files for split '{split_name}'"
            )
        else:
            logger.info(
                f"Collected {len(rows)} audio files for split '{split_name}'"
            )

        # Deterministic order
        rows.sort(key=lambda r: (r.spk_id, r.snt_id))

        _write_csv(rows, csv_file)
        logger.info(f"[{split_name}] {len(rows)} rows -> {csv_file}")

    # Save options
    try:
        save_pkl(conf, os.path.join(save_folder, OPT_FILE))
    except Exception:
        pass

    # Optional merge (could be useful if use of unsupervised/semi supersived data)
    if merge_lst and merge_name:
        csv_to_merge = [os.path.join(save_folder, s + ".csv") for s in merge_lst]
        merged_csv = os.path.join(save_folder, merge_name + ".csv")
        merge_csvs(csv_to_merge, merged_csv)
        logger.info(f"Merged -> {merged_csv}")
