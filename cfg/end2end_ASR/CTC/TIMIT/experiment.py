from speechbrain.core import load_params
functions, params = load_params()

functions.copy_locally([])
functions.prepare_timit([])

# training/validation epochs
functions.training_nn([])

# test
mode='test'
avg_loss, avg_wer = functions.test(mode)
print("Final WER: %f" % (avg_wer))
