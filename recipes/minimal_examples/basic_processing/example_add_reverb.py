import os
import speechbrain as sb

output_folder = os.path.join("results", "add_reverb")
experiment_dir = os.path.dirname(os.path.abspath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")
overrides = {
    "output_folder": output_folder,
    "data_folder": os.path.join(experiment_dir, "..", "..", "..", "samples"),
}
with open(hyperparams_file) as fin:
    hyperparams = sb.load_extended_yaml(fin, overrides)

sb.create_experiment_directory(
    experiment_directory=output_folder,
    hyperparams_to_save=hyperparams_file,
    overrides=overrides,
)

for ((id, wav, wav_len),) in hyperparams["sample_data"]().get_dataloader():
    wav_reverb = hyperparams["add_reverb"](wav, wav_len)
    hyperparams["save"](wav_reverb, id, wav_len)


def test_reverb():
    import torchaudio
    from glob import glob

    for filename in glob(os.path.join(output_folder, "save", "*.wav")):
        expected_file = filename.replace("results", "expected")
        actual, rate = torchaudio.load(filename)
        expected, rate = torchaudio.load(expected_file)
        assert actual.allclose(expected)
