import os
import speechbrain as sb

output_folder = "results/drop_freq"
overrides = {"output_folder": output_folder}
current_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(current_dir, "params.yaml")
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

sb.core.create_experiment_directory(
    experiment_directory=output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

for ((id, wav, wav_len),) in params.sample_data():
    wav_drop = params.drop_freq(wav)
    params.save(wav_drop, id, wav_len)
