import os
import speechbrain as sb

output_folder = os.path.join("results", "do_clip")
experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
overrides = {
    "output_folder": output_folder,
    "data_folder": os.path.join(experiment_dir, "..", "..", "..", "samples"),
}
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

sb.core.create_experiment_directory(
    experiment_directory=output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

for ((id, wav, wav_len),) in params.sample_data():
    wav_clip = params.do_clip(wav)
    params.save(wav_clip, id, wav_len)


def test_clip():
    import torchaudio
    from glob import glob

    for filename in glob(os.path.join(output_folder, "save", "*.wav")):
        expected_file = filename.replace("results", "expected")
        actual, rate = torchaudio.load(filename)
        expected, rate = torchaudio.load(expected_file)
        assert actual.allclose(expected)
