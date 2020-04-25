import sys
from speechbrain.core import Experiment

overrides = {"constants": {"task": "add_reverb"}}
with open("recipes/minimal_examples/basic_processing/params.yaml") as fin:
    sb = Experiment(fin, overrides, commandline_args=sys.argv[1:])

for (batch,) in zip(*sb.sample_data()):
    id, wav, wav_len = batch
    wav_reverb = sb.add_reverb(wav, wav_len)
    sb.save(wav_reverb, id, wav_len)
