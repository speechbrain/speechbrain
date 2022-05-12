"""Example recipe to benchmark SpeechBrain using PyTorch profiling.

A pretrained interference is benchmarked for real-time factors and memory peaks across audio durations and batch sizes.
Profiling is carried out either on random data (pure noise) or on an example file that is truncated and repeated to
be representative of a benchmark setting (duration vs batch size). The setup is defined in: profile.yaml.
@profile_optimiser is used: the last two/six batches are recorded for profiling.

Run from within this directory (yaml defines an example audio w/ relative path):
`python profile.py profile.yaml`

Operational for: EncoderDecoderASR; EncoderASR

Author:
    * Andreas Nautsch 2022
"""
import sys
import torch
import pandas
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.profiling import (
    profile_optimiser,
    report_time,
    report_memory,
)
from speechbrain.pretrained import (
    Pretrained,
    EncoderDecoderASR,
    EncoderASR,
    EndToEndSLU,
    EncoderClassifier,
    SpeakerRecognition,
    VAD,
    SepformerSeparation,
    SpectralMaskEnhancement,
    SNREstimator,
)
from typing import Optional, Iterable


def prepare_unary_input(
    batch_size,
    duration,
    batch_label="wavs",
    lengths_label: Optional[str] = "wav_lens",
    example=None,
    sampling_rate=16000,
):
    unary_input = {
        batch_label: example[: duration * sampling_rate].repeat(batch_size, 1)
        if example is not None
        else torch.rand((batch_size, duration * sampling_rate)),
    }
    if lengths_label is not None:
        unary_input[lengths_label] = torch.ones(batch_size)
    return unary_input


def get_funcs_to_unary_input_classifier(
    cls,
    call_func: str,
    device: torch.device,
    example_audio=None,
    batch_label="wavs",
    lengths_label: Optional[str] = "wav_lens",
):
    assert issubclass(cls, Pretrained)
    pretrained = cls.from_hparams(
        source=source, savedir=save_dir, run_opts={"device": device}
    )
    if example_audio:
        example = pretrained.load_audio(example_audio)

    def prepare(batch_size, duration, sampling_rate=16000):
        return prepare_unary_input(
            batch_size,
            duration,
            batch_label=batch_label,
            lengths_label=lengths_label,
            example=example,
            sampling_rate=sampling_rate,
        )

    def call(model, **kwargs):
        getattr(model, call_func)(**kwargs)

    return prepare, call, pretrained


def get_funcs_to_profile(
    pretrained_type, source, save_dir, example_audio=None, example=None
):
    """Creates per pretrained interface:

    pretrained - loaded model, to device
    prepare(batch_size, duration, sampling_rate=16000) - function handle to create dimensioned batch input
    call(model, **kwargs) - function handle to the inference function to be profiled
    """

    # Put all data directly to cpu/cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create prepare() and call() functions depending on model type
    if pretrained_type == "EncoderDecoderASR":
        return get_funcs_to_unary_input_classifier(
            cls=EncoderDecoderASR,
            call_func="transcribe_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "EncoderASR":
        return get_funcs_to_unary_input_classifier(
            cls=EncoderASR,
            call_func="transcribe_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "EndToEndSLU":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=EndToEndSLU,
            call_func="decode_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "EncoderClassifier":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=EncoderClassifier,
            call_func="classify_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "SpeakerRecognition":  # untested
        pretrained = SpeakerRecognition.from_hparams(
            source=source, savedir=save_dir, run_opts={"device": device}
        )
        if example_audio:
            example = pretrained.load_audio(example_audio)

        def prepare(batch_size, duration, num_wavs2=10, sampling_rate=16000):
            return {
                "wavs1": torch.rand((batch_size, duration * sampling_rate)),
                "wavs2": torch.rand((num_wavs2, duration * sampling_rate)),
                "wav1_lens": torch.ones(batch_size),
                "wav2_lens": torch.ones(num_wavs2),
            }

        def call(model, **kwargs):
            model.verify_batch(**kwargs)

    elif pretrained_type == "VAD":  # untested
        # VAD boundary post-processing can introduce slightly more load (ignored here)
        return get_funcs_to_unary_input_classifier(
            cls=VAD,
            call_func="get_speech_prob_chunk",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "SepformerSeparation":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=SepformerSeparation,
            call_func="separate_batch",
            example_audio=example_audio,
            device=device,
            batch_label="mix",
            lengths_label=None,
        )

    elif pretrained_type == "SpectralMaskEnhancement":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=SpectralMaskEnhancement,
            call_func="enhance_batch",
            example_audio=example_audio,
            device=device,
            batch_label="noisy",
            lengths_label="lengths",
        )

    elif pretrained_type == "SNREstimator":  # untested
        pretrained = SNREstimator.from_hparams(
            source=source, savedir=save_dir, run_opts={"device": device}
        )
        if example_audio:
            example = pretrained.load_audio(example_audio)

        def prepare(batch_size, duration, num_spks=2, sampling_rate=16000):
            return {
                "mix": example[: duration * sampling_rate].repeat(batch_size, 1)
                if example is not None
                else torch.rand((batch_size, duration * sampling_rate)),
                "predictions": torch.rand(
                    (batch_size, duration * sampling_rate, num_spks)
                ),
            }

        def call(model, **kwargs):
            model.estimate_batch(**kwargs)

    else:  # pretrained_type must be part of SpeechBrain
        raise TypeError("Unknown pretrained model.")

    return prepare, call, pretrained


def profile_pretrained(
    pretrained_type,
    source,
    save_dir,
    audio_mockup_secs,
    batch_sizes,
    triangle_only=True,
    example_audio=None,
    export_logs=False,
    output_format: Optional[Iterable] = "markdown",
    ext: Optional[Iterable] = "md",
):
    """Loops through the profiler settings and benchmarks the inference of a pretrained model.

    Reporting:
    - real time factor
    - peak memory (inference only)

    Logs:
    - shell w/ tabular profiler summary and targeted reporting
    - if export_logs: traces are stored in `log` folder
    - benchmark_real_time (pandas.DataFrame)
    - memory_peaks (pandas.DataFrame)
    """
    # Pretrained interface
    create_batch_data, call, pretrained = get_funcs_to_profile(
        pretrained_type, source, save_dir, example_audio
    )

    # Prepare table to write out profiling information
    realtime_factor = pandas.DataFrame(
        index=audio_mockup_secs, columns=batch_sizes
    )
    memory_peaks = pandas.DataFrame(
        index=audio_mockup_secs, columns=batch_sizes
    )
    us_in_s = 1000.0 ** 2
    byte_in_GB = 1024.0 ** 3

    # Comprehensive benchmarking
    for d, duration in enumerate(audio_mockup_secs):
        for b, bs in enumerate(batch_sizes):
            # skip expected heavy-loads
            if (
                triangle_only
            ):  # this is a protection mechanism, since configs might explore exponentially
                if (
                    (b + d >= (len(audio_mockup_secs) + len(batch_sizes)) / 2)
                    and (d > 0)
                    and (b > 0)
                ):
                    print(
                        f"\tskipped - duration: {duration:d}, batch_size: {bs:d}"
                    )
                    realtime_factor.loc[duration, bs] = "N/A"
                    memory_peaks.loc[duration, bs] = "N/A"
                    continue

            # where are we :)
            print(f"\nDuration: {duration:d}, batch_size: {bs:d}")

            # benchmarking
            kwargs = create_batch_data(batch_size=bs, duration=duration)
            realtime = (
                2 * bs * us_in_s * duration
            )  # 2 batches recorded x conversion factor x secs

            # Simulating batching and profiling it
            with profile_optimiser(export_logs=export_logs) as prof:
                for _ in range(
                    6
                ):  # default scheduler records in fifth and sixth step
                    call(model=pretrained, **kwargs)
                    prof.step()

            # Gathering time and memory reports
            print(
                prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10
                )
            )
            cpu_time, cuda_time = report_time(prof)
            cpu_mem, cuda_mem = report_memory(prof, verbose=True)

            # Keep formatted figures for tables
            cpu_max = cpu_mem.max()
            cuda_max = cuda_mem.max()

            if cuda_time == 0:  # CPU values only
                realtime_factor.loc[duration, bs] = f"{cpu_time / realtime:.2E}"
                memory_peaks.loc[
                    duration, bs
                ] = f"{cpu_max / byte_in_GB:.2f} Gb"
            else:  # CPU | GPU
                realtime_factor.loc[
                    duration, bs
                ] = f"{cpu_time / realtime:.2E} + {cuda_time / realtime:.2E}"
                memory_peaks.loc[
                    duration, bs
                ] = f"{cpu_max / byte_in_GB:.2f} + {cuda_max / byte_in_GB:.2f} Gb"

    # Store tables
    print("\n\tReal-time factor")
    print(realtime_factor)
    print("\n\tPeak memory")
    print(memory_peaks)

    if output_format is not None:
        for of, e in zip(output_format, ext):
            getattr(realtime_factor, "to_%s" % of)("benchmark_real_time.%s" % e)
            getattr(memory_peaks, "to_%s" % of)("benchmark_memory.%s" % e)


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides, overrides_must_match=False)

    # Ensure profiling dimensions are set
    profiling_setup = {
        "audio_mockup_secs": [1, 2],
        "batch_sizes": [1, 2],
        "triangle_only": True,
        "example_audio": None,
        "export_logs": False,
    }
    if "profiling_dimensions" in hparams:
        if isinstance(hparams["profiling_dimensions"], dict):
            for arg, specification in hparams["profiling_dimensions"].items():
                if arg in profiling_setup:
                    profiling_setup[arg] = specification

    # Lookup on pretrained model and its local storage
    pretrained_type = hparams["pretrained_model"]["type"]
    source = hparams["pretrained_model"]["source"]
    save_dir = f"pretrained_models/{source}"

    # Lookup output formats and determine depending file extensions
    ext = (
        []
    )  # a list to be filled depending on the output formats specified in hparams
    output_format = None  # default: None - unless hparams defines one or more formats (string or a list of strings)
    pandas_format_ext = {
        "csv": "csv",  # key: should match a pandas.to_{key} function - value: a common extension
        "json": "json",
        "latex": "tex",
        "markdown": "md",  # pip install tabulate
        "pickle": "pkl",
    }
    if "output_formats" in hparams:
        if isinstance(hparams["output_formats"], str):
            output_format = [hparams["output_formats"]]
        if isinstance(hparams["output_formats"], list):
            output_format = hparams["output_formats"]
    for of in output_format:
        if of not in pandas_format_ext:
            AssertionError("Unknown output format.")
        else:
            ext.append(pandas_format_ext[of])

    profile_pretrained(
        pretrained_type=pretrained_type,
        source=source,
        save_dir=save_dir,
        output_format=output_format,
        ext=ext,
        **profiling_setup,
    )
