# adapted from https://github.com/speechbrain/speechbrain/blob/master/recipes/minimal_examples/neural_networks/ASR_seq2seq/example_asr_seq2seq_experiment.py
import os
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.decoders.decoders import undo_padding
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import torch

output_folder = os.path.join("results", "g2p")
experiment_dir = os.path.dirname(os.path.abspath(__file__))
params_file = os.path.join(experiment_dir, "hparams.yaml")
overrides = {
}

with open(params_file) as fin:
	hparams = sb.yaml.load_extended_yaml(fin, overrides)

sb.core.create_experiment_directory(
	experiment_directory=output_folder,
	params_to_save=params_file,
)

searcher = S2SRNNGreedySearcher(
    modules=[
        hparams.decoder_embed,
        hparams.decoder_net,
        hparams.decoder_linear,
        hparams.logsoftmax,
    ],
    bos_index=hparams.bos,
    eos_index=hparams.eos,
    min_decode_ratio=0,
    max_decode_ratio=10.0,
)

# Create train.csv, dev.csv, test.csv from lexicon.csv

import pandas as pd
from sklearn.model_selection import train_test_split
lexicon = pd.read_csv(hparams.input_lexicon)
train_lexicon, other = train_test_split(lexicon, test_size=0.20, random_state=hparams.seed)
valid_lexicon, test_lexicon = train_test_split(other, test_size=0.50, random_state=hparams.seed)
train_lexicon.to_csv(hparams.csv_train, index=False)
valid_lexicon.to_csv(hparams.csv_valid, index=False)
test_lexicon.to_csv(hparams.csv_test, index=False)

class G2P(sb.core.Brain):
	def compute_forward(self, x, y, stage="train", init_params=False):
		_, graphemes, graphemes_lens = x
		_, phonemes, phonemes_lens = y
		phonemes = prepend_bos_token(phonemes, bos_index=hparams.bos)

		x_embedded = hparams.encoder_embed(graphemes.long(), init_params=init_params)
		y_embedded = hparams.decoder_embed(phonemes, init_params=init_params)
		encoder_out = hparams.encoder_net(x_embedded, init_params=init_params)
		h, w = hparams.decoder_net(y_embedded, encoder_out, graphemes_lens, init_params=init_params)
		logits = hparams.decoder_linear(h, init_params=init_params)
		outputs = hparams.logsoftmax(logits)

		if stage != "train":
			seq, _ = searcher(encoder_out, graphemes_lens)
			######
			print(graphemes[0])
			print(seq[0])
			print(phonemes[0])
			######
			return outputs, seq

		return outputs

	def compute_objectives(self, predictions, targets, stage="train"):
		if stage == "train":
			outputs = predictions
		else:
			outputs, seq = predictions

		ids, phonemes, phonemes_lens = targets

		# Add 1 to lengths for eos token
		abs_length = torch.round(phonemes_lens * phonemes.shape[1])
		phonemes = append_eos_token(
			phonemes, length=abs_length, eos_index=hparams.eos
		)
		rel_length = (abs_length + 1)/phonemes.shape[1]

		loss = hparams.compute_cost(outputs, phonemes, length=rel_length)

		stats = {}
		if stage != "train":
			phonemes = undo_padding(phonemes, phonemes_lens)
			stats["PER"] = wer_details_for_batch(ids, phonemes, seq)
		return loss, stats

	def fit_batch(self, batch):
		inputs, targets = batch
		predictions = self.compute_forward(inputs, targets)
		loss, stats = self.compute_objectives(predictions, targets)
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		stats["loss"] = loss.detach()
		return stats

	def evaluate_batch(self, batch, stage="test"):
		inputs, targets = batch
		out = self.compute_forward(inputs, targets, stage="test")
		loss, stats = self.compute_objectives(out, targets, stage="test")
		stats["loss"] = loss.detach()
		return stats

	def on_epoch_end(self, epoch, train_stats, valid_stats):
		print("Epoch %d complete" % epoch)
		print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
		print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
		print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))


train_set = hparams.train_loader()
first_x, first_y = next(iter(train_set))
#print(len(hparams.train_loader.label_dict["graphemes"]["counts"])) --> 27
#print(len(hparams.train_loader.label_dict["phonemes"]["counts"])) --> 39

# why do I have to list the modules?
model = G2P(
	modules=[
		hparams.encoder_embed,
		hparams.encoder_net,
		hparams.decoder_embed,
		hparams.decoder_net,
		hparams.decoder_linear,
	],
	optimizer=hparams.optimizer,
	first_inputs=[first_x, first_y],
)

model.fit(
    range(hparams.N_epochs), train_set, hparams.valid_loader()
)
test_stats = model.evaluate(hparams.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))

