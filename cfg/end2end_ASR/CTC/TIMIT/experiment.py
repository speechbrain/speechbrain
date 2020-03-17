from speechbrain.core import load_params
f, params = load_params(__file__, 'params.yaml')

f.copy_locally([])
f.prepare_timit([])

# training/validation epochs
f.training_nn([])

# test
mode='test'
avg_loss, avg_wer = f.test(mode)
print("Final WER: %f" % (avg_wer))
