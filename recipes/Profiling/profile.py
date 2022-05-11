"""Example recipe to benchmark SpeechBrain using PyTorch profiling.

A pretrained interference is benchmarked for real-time factors and memory peaks across audio durations and batch sizes.
Profiling is carried out either on random data (pure noise) or on an example file that is truncated and repeated to
be representative of a benchmark setting (duration vs batch size). The setup is defined in: profile.yaml.
@profile_optimiser is used: the last two/six batches are recorded for profiling.

Run from within this directory (yaml defines an example audio w/ relative path):
`python profile.py profile.yaml`

Operational for: EncoderDecoderASR

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
from speechbrain.pretrained import EncoderDecoderASR

"""
from speechbrain.pretrained import (
    EndToEndSLU,
    EncoderASR,
    EncoderClassifier,
    SpeakerRecognition,
    VAD,
    SepformerSeparation,
    SpectralMaskEnhancement,
    SNREstimator,
)
"""


def get_funcs_to_profile(pretrained_type, source, save_dir, example_audio=None):
    """Creates per pretrained interface:

    pretrained - loaded model, to device
    prepare(batch_size, duration, sampling_rate=16000) - function handle to create dimensioned batch input
    call(model, **kwargs) - function handle to the inference function to be profiled
    """

    # Put all data directly to cpu/cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create prepare() and call() functions depending on model type
    if pretrained_type == "EncoderDecoderASR":
        pretrained = EncoderDecoderASR.from_hparams(
            source=source, savedir=save_dir, run_opts={"device": device}
        )
        if example_audio:
            example = pretrained.load_audio(example_audio)

        def prepare(batch_size, duration, sampling_rate=16000):
            return {
                "wavs": example[: duration * sampling_rate].repeat(
                    batch_size, 1
                )
                if example_audio
                else torch.rand((batch_size, duration * sampling_rate)),
                "wav_lens": torch.ones(batch_size),
            }

        def call(model, **kwargs):
            model.transcribe_batch(**kwargs)

        """
        elif pretrained_type == "EndToEndSLU":  # untested
            pretrained = EndToEndSLU.from_hparams(source=source, savedir=save_dir)

            def prepare(batch_size, duration, sampling_rate=16000):
                return {
                    "wavs": torch.rand((batch_size, duration * sampling_rate)),
                    "wav_lens": torch.ones(batch_size).to(device),
                }

            def call(model, **kwargs):
                model.decode_batch(**kwargs)

        elif pretrained_type == "EncoderASR":  # untested
            pretrained = EncoderASR.from_hparams(source=source, savedir=save_dir)

            def prepare(batch_size, duration, sampling_rate=16000):
                return {
                    "wavs": torch.rand((batch_size, duration * sampling_rate)),
                    "wav_lens": torch.ones(batch_size),
                }

            def call(model, **kwargs):
                model.transcribe_batch(**kwargs)

        elif pretrained_type == "EncoderClassifier":  # untested
            pretrained = EncoderClassifier.from_hparams(
                source=source, savedir=save_dir
            )

            def prepare(batch_size, duration, sampling_rate=16000):
                return {
                    "wavs": torch.rand((batch_size, duration * sampling_rate)),
                    "wav_lens": torch.ones(batch_size),
                }

            def call(model, **kwargs):
                model.classify_batch(**kwargs)

        elif pretrained_type == "SpeakerRecognition":  # untested
            pretrained = SpeakerRecognition.from_hparams(
                source=source, savedir=save_dir
            )

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
            pretrained = VAD.from_hparams(source=source, savedir=save_dir)

            def prepare(batch_size, duration, sampling_rate=16000):
                return {
                    "wavs": torch.rand((batch_size, duration * sampling_rate)),
                    "wav_lens": torch.ones(batch_size),
                }

            def call(model, **kwargs):
                # VAD boundary post-processing can introduce slightly more load (ignored here)
                model.get_speech_prob_chunk(**kwargs)

        elif pretrained_type == "SepformerSeparation":  # untested
            pretrained = SepformerSeparation.from_hparams(
                source=source, savedir=save_dir
            )

            def prepare(batch_size, duration, sampling_rate=16000):
                return {
                    "mix": torch.rand((batch_size, duration * sampling_rate)),
                }

            def call(model, **kwargs):
                model.separate_batch(**kwargs)

        elif pretrained_type == "SpectralMaskEnhancement":  # untested
            pretrained = SpectralMaskEnhancement.from_hparams(
                source=source, savedir=save_dir
            )

            def prepare(batch_size, duration, sampling_rate=16000):
                return {
                    "noisy": torch.rand((batch_size, duration * sampling_rate)),
                    "lengths": torch.ones(batch_size),
                }

            def call(model, **kwargs):
                model.enhance_batch(**kwargs)

        elif pretrained_type == "SNREstimator":  # untested
            pretrained = SNREstimator.from_hparams(source=source, savedir=save_dir)

            def prepare(batch_size, duration, num_spks=2, sampling_rate=16000):
                return {
                    "mix": torch.rand((batch_size, duration * sampling_rate)),
                    "predictions": torch.rand(
                        (batch_size, duration * sampling_rate, num_spks)
                    ),
                }

            def call(model, **kwargs):
                model.estimate_batch(**kwargs)
        """
    else:  # pretrained_type must be part of SpeechBrain
        raise TypeError("Unknown pretrained model.")

    return prepare, call, pretrained


def profile_pretrained(
    pretrained_type,
    source,
    save_dir,
    audio_mockup_secs,
    batch_sizes,
    upper_triangule_only=True,
    example_audio=None,
    export_logs=False,
    output_format="markdown",
    ext="md",
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
                upper_triangule_only
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
            try:
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
                    realtime_factor.loc[
                        duration, bs
                    ] = f"{cpu_time / realtime:.2E}"
                    memory_peaks.loc[
                        duration, bs
                    ] = f"{cpu_max / byte_in_GB:.2f} Gb"
                else:  # CPU | GPU
                    realtime_factor.loc[
                        duration, bs
                    ] = f"{cpu_time / realtime:.2E} | {cuda_time / realtime:.2E}"
                    memory_peaks.loc[
                        duration, bs
                    ] = f"{cpu_max / byte_in_GB:.2f} | {cuda_max / byte_in_GB:.2f} Gb"
            except Exception as exception:  # if it's out-of-memory, this one will not help ;-)
                print("Error occurred.")
                print(exception)
                realtime_factor.loc[duration, bs] = "N/A"
                memory_peaks.loc[duration, bs] = "N/A"

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
        "upper_triangule_only": True,
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
