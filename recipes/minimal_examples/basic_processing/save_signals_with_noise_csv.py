from speechbrain.core import Experiment

filename = "recipes/minimal_examples/basic_processing/params.yaml"
sb = Experiment(open(filename))

for (batch,) in zip(*sb.sample_data()):
    id, wav, wav_len = batch
    wav_noise = sb.add_noise_csv(wav, wav_len)
    sb.save(wav_noise, id, wav_len)
