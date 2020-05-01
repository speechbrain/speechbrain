import speechbrain as sb

output_folder = "exp/minimal/drop_freq"
overrides = {"output_folder": output_folder}
params_file = "recipes/minimal_examples/basic_processing/params.yaml"
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

sb.core.create_experiment_directory(
    experiment_directory=output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

for ((id, wav, wav_len),) in zip(*params.sample_data()):
    wav_drop = params.drop_freq(wav)
    params.save(wav_drop, id, wav_len)
