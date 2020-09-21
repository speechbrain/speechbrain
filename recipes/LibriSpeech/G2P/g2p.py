# adapted from https://github.com/speechbrain/speechbrain/blob/master/recipes/minimal_examples/neural_networks/ASR_seq2seq/example_asr_seq2seq_experiment.py
import os
import sys
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.decoders import undo_padding
from speechbrain.decoders.seq2seq import S2SRNNBeamSearcher
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_error_rate
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from librispeech_prepare import prepare_librispeech  # noqa E402

params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    hparams = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=hparams.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)

modules = torch.nn.ModuleList(
    [
        hparams.encoder_embed,
        hparams.encoder_net,
        hparams.decoder_embed,
        hparams.decoder_net,
        hparams.decoder_linear,
    ]
)

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
    # the output may be longer than the input, e.g. "b o x" --> "B AA K S"
    max_decode_ratio=10.0,
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
    def compute_forward(self, x, y, stage="train", init_params=False):
        _, graphemes, graphemes_lens = x
        _, phonemes, phonemes_lens = y

        graphemes, graphemes_lens = (
            graphemes.to(hparams.device),
            graphemes_lens.to(hparams.device),
        )
        phonemes, phonemes_lens = (
            phonemes.to(hparams.device),
            phonemes_lens.to(hparams.device),
        )

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
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, targets, stage="train"):
        if stage == "train":
            outputs = predictions
        else:
            outputs, seq = predictions

        ids, phonemes, phonemes_lens = targets
        phonemes, phonemes_lens = (
            phonemes.to(hparams.device),
            phonemes_lens.to(hparams.device),
        )

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
            phonemes = undo_padding(phonemes.long(), phonemes_lens)
            ind2lab = hparams.train_loader.label_dict["phonemes"]["index2lab"]
            phonemes = convert_index_to_lab(phonemes, ind2lab)
            seq = convert_index_to_lab(seq, ind2lab)
            per_stats = wer_details_for_batch(
                ids, phonemes, seq, compute_alignments=True
            )
            stats["PER"] = per_stats
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
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = hparams.lr_annealing([hparams.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        hparams.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        checkpointer.save_and_keep_only(meta={"PER": per}, min_keys=["PER"])


# Prepare data
prepare_librispeech(
    data_folder=hparams.data_folder,
    splits=[],
    save_folder=hparams.data_folder,
    create_lexicon=True,
)


train_set = hparams.train_loader()
valid_set = hparams.valid_loader()
test_set = hparams.test_loader()
first_x, first_y = next(iter(train_set))

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

checkpointer.recover_if_possible()
model.fit(hparams.epoch_counter, train_set, valid_set)

checkpointer.recover_if_possible(min_key="PER")
test_stats = model.evaluate(hparams.test_loader())

hparams.train_logger.log_stats(
    stats_meta={"Epoch loaded": hparams.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(hparams.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
