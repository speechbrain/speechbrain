import sys
from speechbrain.core import Experiment

overrides = {
    "constants": {"task": "add_noise"},
    "functions": {"add_noise": {"snr_low": 0, "snr_high": 0}},
}
with open("recipes/minimal_examples/basic_processing/params.yaml") as fin:
    sb = Experiment(fin, overrides, commandline_args=sys.argv[1:])

_, first_batch, _ = next(iter(*sb.sample_data()))
sb.add_noise.init_params(first_batch)
for (batch,) in zip(*sb.sample_data()):
    id, wav, wav_len = batch
    wav_noise = sb.add_noise(wav, wav_len)
    sb.save(wav_noise, id, wav_len)
