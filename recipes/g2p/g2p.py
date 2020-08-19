# adapted from https://github.com/speechbrain/speechbrain/blob/master/recipes/minimal_examples/neural_networks/ASR_seq2seq/example_asr_seq2seq_experiment.py
import os
import sys
import pandas as pd
import speechbrain as sb
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.decoders.decoders import undo_padding
from speechbrain.decoders.seq2seq import S2SRNNBeamSearcher
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    hparams = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=hparams.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)

modules = torch.nn.ModuleList([
    hparams.encoder_embed,
    hparams.encoder_net,
    hparams.decoder_embed,
    hparams.decoder_net,
    hparams.decoder_linear,
])

searcher = S2SRNNBeamSearcher(
    modules=[
        hparams.decoder_embed,
        hparams.decoder_net,
        hparams.decoder_linear,
        hparams.logsoftmax,
    ],
    bos_index=hparams.bos,
    eos_index=hparams.eos,
    min_decode_ratio=0,
    max_decode_ratio=10.0, # the output may be longer than the input, e.g. "b o x" --> "B AA K S"
    beam_size=10,
)

checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=hparams.save_folder,
    recoverables={
        "model": modules,
        "optimizer": hparams.optimizer,
        "scheduler": hparams.lr_annealing,
        "counter": hparams.epoch_counter,
    },
)

class G2P(sb.core.Brain):
    def decode_batch(self, batch, init_params=False):
        x = batch[0]
        _, graphemes, graphemes_lens = x
        x_embedded = hparams.encoder_embed(
            graphemes.long(), init_params=init_params
        )
        encoder_out = hparams.encoder_net(x_embedded, init_params=init_params)
        seq, _ = searcher(encoder_out, graphemes_lens)
        gg = torch.round(graphemes.shape[1] * graphemes_lens).long()
        for i in range(len(graphemes)):
            # remove padding from search results
            if 0 in seq[i]:
                seq_no_padding = []
                for j in range(len(seq[i])):
                    if seq[i][j] != 0: seq_no_padding += [seq[i][j]]
                    else: break
                seq[i] = seq_no_padding

        # Convert indices to labels
        batch_size = len(graphemes)
        model_in = [graphemes[i][:gg[i]].long() for i in range(batch_size)]
        decoded_inputs = [" ".join([hparams.train_loader.label_dict["graphemes"]["index2lab"][l.item()] for l in s_in]) for s_in in model_in]
        decoded_outputs = [" ".join([hparams.train_loader.label_dict["phonemes"]["index2lab"][l] for l in s_out]) for s_out in seq]

        return decoded_inputs, decoded_outputs

    def compute_forward(self, x, y, stage="train", init_params=False):
        _, graphemes, graphemes_lens = x
        _, phonemes, phonemes_lens = y
        phonemes = prepend_bos_token(phonemes, bos_index=hparams.bos)

        x_embedded = hparams.encoder_embed(
            graphemes.long(), init_params=init_params
        )
        y_embedded = hparams.decoder_embed(phonemes, init_params=init_params)
        encoder_out = hparams.encoder_net(x_embedded, init_params=init_params)
        h, w = hparams.decoder_net(
            y_embedded, encoder_out, graphemes_lens, init_params=init_params
        )
        logits = hparams.decoder_linear(h, init_params=init_params)
        outputs = hparams.logsoftmax(logits)

        if stage != "train":
            seq, _ = searcher(encoder_out, graphemes_lens)
            gg = torch.round(graphemes.shape[1] * graphemes_lens).long()
            for i in range(len(graphemes)):
                # remove padding from search results
                if 0 in seq[i]:
                    seq_no_padding = []
                    for j in range(len(seq[i])):
                        if seq[i][j] != 0: seq_no_padding += [seq[i][j]]
                        else: break
                    seq[i] = seq_no_padding
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
        phonemes_with_eos = append_eos_token(
            phonemes, length=abs_length, eos_index=hparams.eos
        )
        rel_length = (abs_length + 1) / phonemes_with_eos.shape[1]

        loss = hparams.compute_cost(
            outputs, phonemes_with_eos, length=rel_length
        )

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

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        out = self.compute_forward(inputs, targets, stage=stage)
        loss, stats = self.compute_objectives(out, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = hparams.lr_annealing([hparams.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        hparams.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        checkpointer.save_and_keep_only(meta={"PER": per}, min_keys=["PER"])

train_set = hparams.train_loader()
valid_set = hparams.valid_loader()
test_set = hparams.test_loader()
first_x, first_y = next(iter(train_set))

# Add 1 to labels, so that the token for padding =\= the token for a label
temp = {}
for key in hparams.train_loader.label_dict["graphemes"]["index2lab"]:
        index = hparams.train_loader.label_dict["graphemes"]["index2lab"][key]
        temp[key + 1] = index
        hparams.train_loader.label_dict["graphemes"]["lab2index"][index] = key + 1
hparams.train_loader.label_dict["graphemes"]["index2lab"] = temp
hparams.valid_loader.label_dict["graphemes"]["index2lab"] = hparams.train_loader.label_dict["graphemes"]["index2lab"]
hparams.valid_loader.label_dict["graphemes"]["lab2index"] = hparams.train_loader.label_dict["graphemes"]["lab2index"]
hparams.test_loader.label_dict["graphemes"]["index2lab"] = hparams.train_loader.label_dict["graphemes"]["index2lab"]
hparams.test_loader.label_dict["graphemes"]["lab2index"] = hparams.train_loader.label_dict["graphemes"]["lab2index"]

temp = {}
for key in hparams.train_loader.label_dict["phonemes"]["index2lab"]:
        index = hparams.train_loader.label_dict["phonemes"]["index2lab"][key]
        temp[key + 1] = index
        hparams.train_loader.label_dict["phonemes"]["lab2index"][index] = key + 1
hparams.train_loader.label_dict["phonemes"]["index2lab"] = temp
hparams.valid_loader.label_dict["phonemes"]["index2lab"] = hparams.train_loader.label_dict["phonemes"]["index2lab"]
hparams.valid_loader.label_dict["phonemes"]["lab2index"] = hparams.train_loader.label_dict["phonemes"]["lab2index"]
hparams.test_loader.label_dict["phonemes"]["index2lab"] = hparams.train_loader.label_dict["phonemes"]["index2lab"]
hparams.test_loader.label_dict["phonemes"]["lab2index"] = hparams.train_loader.label_dict["phonemes"]["lab2index"]

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

model.fit(range(hparams.N_epochs), train_set, valid_set)

test_stats = model.evaluate(hparams.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))

# Get pronunciations for OOV words; add to lexicon_augmented.
# (As in the other dataloaders, we need to change the labels to deal with padding.)
oov_set = hparams.oov_loader()
hparams.oov_loader.label_dict["graphemes"]["index2lab"] = hparams.train_loader.label_dict["graphemes"]["index2lab"]
hparams.oov_loader.label_dict["graphemes"]["lab2index"] = hparams.train_loader.label_dict["graphemes"]["lab2index"]
ID = []
duration = []
graphemes = []
graphemes_format = []
graphemes_opts = []
phonemes = []
phonemes_format = []
phonemes_opts = []
lexicon = pd.read_csv(hparams.input_lexicon)
current_ID = max(lexicon.ID) + 1
for batch in oov_set:
	decoded_inputs, decoded_outputs = model.decode_batch(batch)
	batch_size = len(decoded_inputs)
	ID += [i for i in range(current_ID, current_ID + batch_size)]
	duration += [len(d.split()) for d in decoded_inputs]
	graphemes += decoded_inputs
	graphemes_format += ["string"] * batch_size
	graphemes_opts += [np.nan] * batch_size
	phonemes += decoded_outputs
	phonemes_format += ["string"] * batch_size
	phonemes_opts += [np.nan] * batch_size
	current_ID += batch_size

augment = pd.DataFrame({
	"ID" : ID,
	"duration" : duration,
	"graphemes" : graphemes,
	"graphemes_format" : graphemes_format,
	"graphemes_opts" : graphemes_opts,
	"phonemes" : phonemes,
	"phonemes_format" : phonemes_format,
	"phonemes_opts" : phonemes_opts
})

lexicon_augmented = pd.concat([lexicon, augment])
lexicon_augmented.to_csv(hparams.output_lexicon, index=False)
print("New lexicon generated (%s)." % hparams.output_lexicon)
