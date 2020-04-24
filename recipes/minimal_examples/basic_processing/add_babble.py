import sys
from speechbrain.core import Experiment

overrides = {
    "constants": {"task": "add_babble", "batch_size": 5},
    "functions": {"add_babble": {"snr_low": 0, "snr_high": 0}},
}
with open("recipes/minimal_examples/basic_processing/params.yaml") as fin:
    sb = Experiment(fin, overrides, commandline_args=sys.argv[1:])

for (batch,) in zip(*sb.sample_data()):
    id, wav, wav_len = batch
    wav_babble = sb.add_babble(wav, wav_len)
    sb.save(wav_babble, id, wav_len)
