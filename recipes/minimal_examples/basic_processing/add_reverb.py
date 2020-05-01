import speechbrain as sb

output_folder = "exp/minimal/add_reverb"
overrides = {"batch_size": 1, "output_folder": output_folder}
params_file = "recipes/minimal_examples/basic_processing/params.yaml"
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

sb.core.create_experiment_directory(
    experiment_directory=output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

for ((id, wav, wav_len),) in zip(*params.sample_data()):
    wav_reverb = params.add_reverb(wav, wav_len)
    params.save(wav_reverb, id, wav_len)
