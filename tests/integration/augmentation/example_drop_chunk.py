import os

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import read_audio, write_audio

experiment_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(experiment_dir, "results", "drop_chunk")
save_folder = os.path.join(output_folder, "save")
hyperparams_file = os.path.join(experiment_dir, "hyperparams.yaml")


def main():
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    overrides = {
        "output_folder": output_folder,
        "data_folder": os.path.join(experiment_dir, "..", "..", "samples"),
    }
    with open(hyperparams_file, encoding="utf-8") as fin:
        hyperparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=output_folder,
        hyperparams_to_save=hyperparams_file,
        overrides=overrides,
    )

    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset=hyperparams["sample_data"], batch_size=hyperparams["batch_size"]
    )
    for id, (wav, wav_len) in iter(dataloader):
        wav_drop = hyperparams["drop_chunk"](wav, wav_len)
        # save results in file
        for i, snt_id in enumerate(id):
            filepath = os.path.join(save_folder, f"{snt_id}.flac")
            write_audio(filepath, wav_drop[i], 16000)


def test_drop_chunk():
    from glob import glob

    main()
    for filename in glob(os.path.join(output_folder, "save", "*.flac")):
        expected_file = filename.replace("results", "expected")

        actual_file_abs = os.path.join(experiment_dir, filename)
        expected_file_abs = os.path.join(experiment_dir, expected_file)

        actual = read_audio(actual_file_abs)
        expected = read_audio(expected_file_abs)
        assert actual.allclose(expected, atol=1e-4)


if __name__ == "__main__":
    test_drop_chunk()
