import os
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import read_audio, write_audio

output_folder = os.path.join("results", "drop_freq")
experiment_dir = os.path.dirname(os.path.abspath(__file__))
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")


def main():
    overrides = {
        "output_folder": output_folder,
        "data_folder": os.path.join(
            experiment_dir, "..", "..", "..", "samples"
        ),
    }
    with open(hyperparams_file) as fin:
        hyperparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=output_folder,
        hyperparams_to_save=hyperparams_file,
        overrides=overrides,
    )

    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset=hyperparams["sample_data"], batch_size=hyperparams["batch_size"]
    )
    for (id, (wav, wav_len),) in iter(dataloader):
        wav_drop = hyperparams["drop_freq"](wav)
        # save results on file
        for i, snt_id in enumerate(id):
            filepath = (
                hyperparams["output_folder"] + "/save/" + snt_id + ".flac"
            )
            write_audio(filepath, wav_drop[i], 16000)


def test_drop_freq():
    from glob import glob

    for filename in glob(os.path.join(output_folder, "save", "*.flac")):
        expected_file = filename.replace("results", "expected")
        actual = read_audio(filename)
        expected = read_audio(expected_file)
        assert actual.allclose(expected)


if __name__ == "__main__":
    main()
