"""Example recipe to benchmark SpeechBrain using PyTorch profiling.

A pretrained interference is benchmarked for real-time factors and memory peaks across audio durations and batch sizes.
Profiling is carried out either on random data (pure noise) or on an example file that is truncated and repeated to
be representative of a benchmark setting (duration vs batch size). The setup is defined in: profile.yaml.
@profile_optimiser is used: the last two/six batches are recorded for profiling.

Run from within this directory (yaml defines an example audio w/ relative path):
`python profile.py profile.yaml`

Author:
    * Andreas Nautsch 2022
"""
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.profiling import (
    profile_report,
    export,
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
from typing import Optional, List


def get_funcs_to_unary_input_classifier(
    cls,
    call_func: str,
    source: str,
    save_dir: str,
    device: torch.device,
    example_audio=None,
    batch_label="wavs",
    lengths_label: Optional[str] = "wav_lens",
):
    """Implement get_funcs_to_unary_input_classifier."""
    assert issubclass(cls, Pretrained)
    pretrained = cls.from_hparams(
        source=source, savedir=save_dir, run_opts={"device": device}
    )
    example = pretrained.load_audio(example_audio) if example_audio else None

    def prepare(batch_size, duration, sampling_rate=16000):
        """Prepares input data."""
        unary_input = {
            batch_label: example[: duration * sampling_rate].repeat(
                batch_size, 1
            )
            if example is not None
            else torch.rand(
                (batch_size, duration * sampling_rate), device=device
            ),
        }
        if lengths_label is not None:
            unary_input[lengths_label] = torch.ones(batch_size)
        return unary_input

    def call(model, **kwargs):
        """Calls the specified funnction."""
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
            source=source,
            save_dir=save_dir,
            call_func="transcribe_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "EncoderASR":
        return get_funcs_to_unary_input_classifier(
            cls=EncoderASR,
            source=source,
            save_dir=save_dir,
            call_func="transcribe_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "EndToEndSLU":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=EndToEndSLU,
            source=source,
            save_dir=save_dir,
            call_func="decode_batch",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "EncoderClassifier":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=EncoderClassifier,
            source=source,
            save_dir=save_dir,
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
            """Prepares input data."""
            return {
                "wavs1": torch.rand(
                    (batch_size, duration * sampling_rate), device=device
                ),
                "wavs2": torch.rand(
                    (num_wavs2, duration * sampling_rate), device=device
                ),
                "wav1_lens": torch.ones(batch_size),
                "wav2_lens": torch.ones(num_wavs2),
            }

        def call(model, **kwargs):
            """Calls verify_batch."""
            model.verify_batch(**kwargs)

    elif pretrained_type == "VAD":  # untested
        # VAD boundary post-processing can introduce slightly more load (ignored here)
        return get_funcs_to_unary_input_classifier(
            cls=VAD,
            source=source,
            save_dir=save_dir,
            call_func="get_speech_prob_chunk",
            example_audio=example_audio,
            device=device,
        )

    elif pretrained_type == "SepformerSeparation":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=SepformerSeparation,
            source=source,
            save_dir=save_dir,
            call_func="separate_batch",
            example_audio=example_audio,
            device=device,
            batch_label="mix",
            lengths_label=None,
        )

    elif pretrained_type == "SpectralMaskEnhancement":  # untested
        return get_funcs_to_unary_input_classifier(
            cls=SpectralMaskEnhancement,
            source=source,
            save_dir=save_dir,
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
            """Prepares input data."""
            return {
                "mix": example[: duration * sampling_rate].repeat(batch_size, 1)
                if example is not None
                else torch.rand(
                    (batch_size, duration * sampling_rate), device=device
                ),
                "predictions": torch.rand(
                    (batch_size, duration * sampling_rate, num_spks),
                    device=device,
                ),
            }

        def call(model, **kwargs):
            """Calls estimate_batch"""
            model.estimate_batch(**kwargs)

    else:  # pretrained_type must be part of SpeechBrain
        raise TypeError("Unknown pretrained model.")

    return prepare, call, pretrained


def benchmark_to_markdown(
    benchmark: List[List[str]], columns: List[str], rows: List[str]
):
    """Implement benchmark to markdown."""
    cell_width = max([len(x) for x in benchmark[0]])
    fmt = "{: >%d} " % cell_width
    out = (
        "|   "
        + fmt.format("|")
        + "| ".join([fmt.format(x) for x in columns])
        + "|\n"
    )
    sep = "|:" + cell_width * "-" + ":"
    out += (1 + len(columns)) * sep + "|\n"
    for i, r in enumerate(rows):
        out += "| " + fmt.format("%ds " % r)
        out += "| " + " | ".join(benchmark[i]) + " |\n"
    print(out)
    return out


def profile_pretrained(
    pretrained_type,
    source,
    save_dir,
    audio_mockup_secs,
    batch_sizes,
    triangle_only=True,
    example_audio=None,
    export_logs=False,
):
    """Loops through the profiler settings and benchmarks the inference of a pretrained model.

    Reporting:
    - real time factor
    - peak memory (inference only)

    Logs:
    - shell w/ tabular profiler summary and targeted reporting
    - if export_logs: traces are stored in `log` folder
    - benchmark_real_time (file output)
    - memory_peaks (file output)
    """
    # Pretrained interface
    create_batch_data, call, pretrained = get_funcs_to_profile(
        pretrained_type, source, save_dir, example_audio
    )

    # Prepare table to write out profiling information
    realtime_factor = []
    memory_peaks = []
    us_in_s = 1000.0 ** 2
    byte_in_GB = 1024.0 ** 3

    # Comprehensive benchmarking
    for d, duration in enumerate(audio_mockup_secs):
        realtime_factor_row = []
        memory_peaks_row = []
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
                    realtime_factor_row.append("_skip_")
                    memory_peaks_row.append("_skip_")
                    continue

            # where are we :)
            print(f"\nDuration: {duration:d}, batch_size: {bs:d}")

            # benchmarking
            kwargs = create_batch_data(batch_size=bs, duration=duration)
            realtime = (
                bs * us_in_s * duration
            )  # batches recorded x conversion factor x secs

            # Simulating batching and profiling it
            prof = export(profile_report()) if export_logs else profile_report()
            num_steps = 10  # profile_report scheduler needs 10 steps for seven recordings
            for _ in range(num_steps):
                call(model=pretrained, **kwargs)
                prof.step()

            # Gathering time and memory reports
            print(
                prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10
                )
            )

            cpu_time, cuda_time = report_time(
                prof, verbose=True, upper_control_limit=True
            )  # no need to avg #records
            cpu_mem, cuda_mem = report_memory(prof, verbose=True)

            if cuda_time == 0:  # CPU values only
                realtime_factor_row.append(f"{cpu_time / realtime:.2E}")
                memory_peaks_row.append(f"{cpu_mem / byte_in_GB:.2f} Gb")
            else:  # CPU + GPU values
                realtime_factor_row.append(
                    f"{cpu_time / realtime:.2E} + {cuda_time / realtime:.2E}"
                )
                memory_peaks_row.append(
                    f"{cpu_mem / byte_in_GB:.2f} + {cuda_mem / byte_in_GB:.2f} Gb"
                )
        realtime_factor.append(realtime_factor_row)
        memory_peaks.append(memory_peaks_row)

    # Store tables
    print("\n\tReal-time factor")
    with open("bechmark_realtime_factors.md", "w") as f:
        f.write(
            benchmark_to_markdown(
                benchmark=realtime_factor,
                columns=batch_sizes,
                rows=audio_mockup_secs,
            )
        )

    print("\n\tPeak memory")
    with open("bechmark_memory_peaks.md", "w") as f:
        f.write(
            benchmark_to_markdown(
                benchmark=memory_peaks,
                columns=batch_sizes,
                rows=audio_mockup_secs,
            )
        )


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

    profile_pretrained(
        pretrained_type=pretrained_type,
        source=source,
        save_dir=save_dir,
        **profiling_setup,
    )
