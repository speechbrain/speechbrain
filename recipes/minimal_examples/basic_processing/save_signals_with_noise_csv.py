from speechbrain.core import load_params
sb, params = load_params('params.yaml')

for (batch,) in zip(*sb.sample_data()):
    id, wav, wav_len = batch
    wav_noise = sb.add_noise_csv(wav, wav_len)
    sb.save(wav_noise, id, wav_len)
