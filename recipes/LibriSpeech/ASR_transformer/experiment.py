#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from functools import partial

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token

from searchs import GreedySearch, BeamSearch
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.lobes.models.transformer.Transformer import (
    get_key_padding_mask,
    get_lookahead_mask,
)

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from librispeech_prepare import prepare_librispeech  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

modules = torch.nn.ModuleList(
    [params.CNN, params.Transformer, params.ctc_lin, params.seq_lin]
)

checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": modules,
        "optimizer": params.optimizer,
        "scheduler": params.lr_annealing,
        "normalizer": params.normalize,
        "counter": params.epoch_counter,
    },
)


# Define a beam search according to this recipe
valid_search = GreedySearch(
    modules=[params.Transformer, params.seq_lin],
    bos_index=params.bos_index,
    eos_index=params.eos_index,
    min_decode_ratio=0,
    max_decode_ratio=1,
)

test_search = BeamSearch(
    modules=[params.Transformer, params.seq_lin],
    bos_index=params.bos_index,
    eos_index=params.eos_index,
    eos_threshold=params.eos_threshold,
    min_decode_ratio=0,
    max_decode_ratio=1,
    beam_size=params.test_beam_size,
    length_penalty=params.length_penalty,
)


def int2lab(batch, ind2lab):
    batch = batch.tolist()
    batch = [[x for x in batch if x != params.pad_id]]
    batch = convert_index_to_lab(batch, ind2lab)
    return " ".join(batch[-1])


def count_and_pad_bpe_outputs(batch, pad_id):
    # get the lengh and maximum length for every input in the batch
    max_len = 0
    seq_lengths = []
    for seq in batch:
        max_len = max(max_len, seq.shape[-1])
        seq_lengths.append(seq.shape[-1])
    seq_lengths = torch.tensor(seq_lengths).float()

    # batching
    padded_batch = []
    for seq in batch:
        seq = seq.long()

        num_padding_elem = max_len - seq.shape[-1]
        paddings = torch.tensor([pad_id] * num_padding_elem).long()

        seq = torch.cat([seq, paddings], dim=-1)
        padded_batch.append(seq.unsqueeze(0))
    return (
        torch.cat(padded_batch, dim=0).to(params.device),
        seq_lengths.to(params.device),
    )


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        ids, chars, phn_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        chars, phn_lens = chars.to(params.device), phn_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)

        # convert words to bpe
        ind2lab = params.train_loader.label_dict["wrd"]["index2lab"]
        chars, _ = params.tokenizer(
            chars,
            int2lab=partial(int2lab, ind2lab=ind2lab),
            task="encode",
            init_params=init_params,
        )
        chars, seq_lengths = count_and_pad_bpe_outputs(chars, params.pad_id)

        # foward the model
        target_chars = chars
        chars = prepend_bos_token(chars, bos_index=params.bos_index)
        src = params.CNN(feats, init_params=init_params)

        # generate attn mask and padding mask for transformer
        src_key_padding_mask = None
        trg_key_padding_mask = get_key_padding_mask(
            chars, pad_idx=params.pad_id
        )
        src_mask = None
        trg_mask = get_lookahead_mask(chars)

        # repeat targe masks n time in the case of multi-gpu
        if params.multigpu and not init_params:
            ngpu = torch.cuda.device_count()
            trg_mask = trg_mask.repeat(ngpu, 1)

        enc_out, pred = params.Transformer(
            src,
            chars,
            src_mask,
            trg_mask,
            src_key_padding_mask,
            trg_key_padding_mask,
            init_params=init_params,
        )

        # output layer for ctc log-probabilities
        logits = params.ctc_lin(src, init_params)
        p_ctc = params.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = params.seq_lin(pred, init_params)
        p_seq = params.log_softmax(pred)

        # initialize transformer's parameter with xavier_normal
        if init_params:
            self._reset_params()

        # share weight betwenn embedding layer and linear projection layer
        if init_params:
            params.seq_lin.w.weight = (
                params.Transformer.custom_tgt_module.emb.Embedding.weight
            )

        if (
            stage == "valid"
            and params.epoch_counter.current % params.num_epoch_to_valid_search
            == 0
        ):
            hyps, _ = valid_search(enc_out, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps, target_chars, seq_lengths

        elif stage == "test":
            hyps, _ = test_search(enc_out, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps, target_chars, seq_lengths

        return p_ctc, p_seq, wav_lens, target_chars, seq_lengths

    def compute_objectives(self, predictions, targets, stage="train"):
        if (
            stage == "valid"
            and params.epoch_counter.current % params.num_epoch_to_valid_search
            == 0
        ) or stage == "test":
            p_ctc, p_seq, wav_lens, hyps, chars, seq_lengths = predictions
        else:
            p_ctc, p_seq, wav_lens, chars, seq_lengths = predictions

        ids, target_chars, target_lens = targets
        chars, char_lens = (
            chars.to(params.device),
            seq_lengths.to(params.device),
        )

        # Add char_lens by one for eos token
        abs_length = char_lens.float()

        # Append eos token at the end of the label sequences
        phns_with_eos = append_eos_token(
            chars, length=abs_length, eos_index=params.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / chars.shape[1]
        char_lens = char_lens / chars.shape[1]

        loss_ctc = params.ctc_cost(p_ctc, chars, wav_lens, char_lens)
        loss_seq = params.seq_cost(p_seq, phns_with_eos, rel_length)
        loss = params.ctc_weight * loss_ctc + (1 - params.ctc_weight) * loss_seq

        stats = {}
        if (
            stage == "valid"
            and params.epoch_counter.current % params.num_epoch_to_valid_search
            == 0
        ) or stage == "test":
            ind2lab = params.train_loader.label_dict["wrd"]["index2lab"]
            char_seq = params.tokenizer(hyps, task="decode")
            print("hypes_idx", hyps)
            print("hyps", char_seq)

            chars = undo_padding(target_chars, target_lens)
            chars = convert_index_to_lab(chars, ind2lab)
            print("targe", chars)

            wer_stats = edit_distance.wer_details_for_batch(
                ids, chars, char_seq, compute_alignments=True
            )
            stats["WER"] = wer_stats
        return loss, stats

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.modules.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()

        # anneal lr every step
        old_lr, new_lr = params.lr_annealing([params.optimizer], None, None)

        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss, stats = self.compute_objectives(predictions, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        old_lr = params.lr_annealing.current_lr
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        checkpointer.save_and_keep_only(
            meta={"loss": valid_stats["loss"][-1].cpu().item()},
            importance_keys=[ckpt_recency, lambda c: -c.meta["loss"]],
        )

    def _reset_params(self):
        for p in params.Transformer.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


# Prepare data
prepare_librispeech(
    data_folder=params.data_folder,
    splits=[params.train_set, "dev-clean", "test-clean"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(iter(train_set))

# add padding token to index2lab
params.train_loader.label_dict["wrd"]["index2lab"][params.pad_id] = "<pad>"

if hasattr(params, "augmentation"):
    modules.append(params.augmentation)
asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)

if params.multigpu:
    params.CNN = torch.nn.DataParallel(params.CNN)
    params.Transformer = torch.nn.DataParallel(params.Transformer)
    params.ctc_lin = torch.nn.DataParallel(params.ctc_lin)
    params.seq_lin = torch.nn.DataParallel(params.seq_lin)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["loss"])
test_stats = asr_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
wer_summary = edit_distance.wer_summary(test_stats["WER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(wer_summary, fo)
    wer_io.print_alignments(test_stats["WER"], fo)
