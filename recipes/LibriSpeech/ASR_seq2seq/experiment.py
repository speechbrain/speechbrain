#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.data_io.data_io import split_word

from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
from speechbrain.decoders.seq2seq import S2SRNNBeamSearcher
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate

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
    hyperparams_to_save=params_file,
    overrides=overrides,
)

modules = torch.nn.ModuleList(
    [params.enc, params.emb, params.dec, params.ctc_lin, params.seq_lin]
)


class MyBeamSearcher(S2SRNNBeamSearcher):
    def lm_forward_step(self, inp_tokens, memory):
        hs = memory
        (model,) = self.lm_modules
        logits, hs = model(inp_tokens, hx=hs, init_params=self.init_lm_params)
        log_probs = params.log_softmax(logits)

        # set it to false after initialization
        if self.init_lm_params:
            self.init_lm_params = False
        return log_probs, hs

    def ctc_forward_step(self, x):
        logits = params.ctc_lin(x, self.init_ctc_params)
        log_probs = params.log_softmax(logits)

        # set it to false after initialization
        if self.init_ctc_params:
            self.init_ctc_params = False
        return log_probs

    def permute_lm_mem(self, memory, index):
        # This is to permute lm memory to synchronize with current index during beam search.
        # The order of beams will be shuffled by scores every timestep to allow batched beam search.
        # Further details please refer to speechbrain/decoder/seq2seq.py.

        if isinstance(memory, tuple):
            memory_0 = torch.index_select(memory[0], dim=1, index=index)
            memory_1 = torch.index_select(memory[1], dim=1, index=index)
            memory = (memory_0, memory_1)
        else:
            memory = torch.index_select(memory, dim=1, index=index)
        return memory

    def reset_lm_mem(self, batch_size, device):
        # set hidden_state=None, pytorch RNN will automatically set it to zero vectors.
        return None


# Beamsearch with external LM
if hasattr(params, "lm_ckpt_file"):

    lm_modules = torch.nn.ModuleList([params.lm_model])
    lm_modules.eval()

    beam_searcher = MyBeamSearcher(
        modules=[params.emb, params.dec, params.seq_lin],
        bos_index=params.bos_index,
        eos_index=params.eos_index,
        min_decode_ratio=0,
        max_decode_ratio=1,
        beam_size=params.beam_size,
        eos_threshold=params.eos_threshold,
        using_max_attn_shift=params.using_max_attn_shift,
        max_attn_shift=params.max_attn_shift,
        lm_weight=params.lm_weight,
        ctc_weight=params.ctc_weight,
        lm_modules=lm_modules,
    )

else:
    # Beamsearch without LM
    beam_searcher = S2SRNNBeamSearcher(
        modules=[params.emb, params.dec, params.seq_lin],
        bos_index=params.bos_index,
        eos_index=params.eos_index,
        min_decode_ratio=0,
        max_decode_ratio=1,
        beam_size=params.beam_size,
        eos_threshold=params.eos_threshold,
        using_max_attn_shift=params.using_max_attn_shift,
        max_attn_shift=params.max_attn_shift,
    )


# Greedy Search (used for validation only)
greedy_searcher = S2SRNNGreedySearcher(
    modules=[params.emb, params.dec, params.seq_lin],
    bos_index=params.bos_index,
    eos_index=params.eos_index,
    min_decode_ratio=0,
    max_decode_ratio=1,
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


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        ids, words, word_lens = y
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        if stage == "train":
            if hasattr(params, "env_corrupt"):
                wavs_noise = params.env_corrupt(wavs, wav_lens, init_params)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                words = torch.cat([words, words], dim=0)
                word_lens = torch.cat([word_lens, word_lens])
            index2lab = params.train_loader.label_dict["wrd"]["index2lab"]
        elif stage == "valid":
            index2lab = params.valid_loader.label_dict["wrd"]["index2lab"]
        elif stage == "test":
            index2lab = params.test_loader.label_dict["wrd"]["index2lab"]
        bpe, _ = params.bpe_tokenizer(
            words, word_lens, index2lab, task="encode", init_params=init_params
        )
        bpe = bpe.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        x = params.enc(feats, init_params=init_params)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(bpe, bos_index=params.bos_index)
        e_in = params.emb(y_in, init_params=init_params)
        h, _ = params.dec(e_in, x, wav_lens, init_params)

        # output layer for seq2seq log-probabilities
        logits = params.seq_lin(h, init_params)
        p_seq = params.log_softmax(logits)

        if (
            stage == "train"
            and params.epoch_counter.current <= params.number_of_ctc_epochs
        ):
            # output layer for ctc log-probabilities
            logits = params.ctc_lin(x, init_params)
            p_ctc = params.log_softmax(logits)
            return p_ctc, p_seq, wav_lens
        elif stage == "train":
            return p_seq, wav_lens
        elif stage == "valid":
            hyps, scores = greedy_searcher(x, wav_lens)
            return p_seq, wav_lens, hyps
        elif stage == "test":
            hyps, scores = beam_searcher(x, wav_lens)
            return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, targets, stage="train"):
        if (
            stage == "train"
            and params.epoch_counter.current <= params.number_of_ctc_epochs
        ):
            index2lab = params.train_loader.label_dict["wrd"]["index2lab"]
            p_ctc, p_seq, wav_lens = predictions
        elif stage == "train":
            index2lab = params.train_loader.label_dict["wrd"]["index2lab"]
            p_seq, wav_lens = predictions
        else:
            if stage == "valid":
                index2lab = params.valid_loader.label_dict["wrd"]["index2lab"]
            else:
                index2lab = params.test_loader.label_dict["wrd"]["index2lab"]
            p_seq, wav_lens, hyps = predictions

        ids, words, word_lens = targets
        bpe, bpe_lens = params.bpe_tokenizer(
            words, word_lens, index2lab, task="encode"
        )
        bpe, bpe_lens = bpe.to(params.device), bpe_lens.to(params.device)
        if hasattr(params, "env_corrupt") and stage == "train":
            bpe = torch.cat([bpe, bpe], dim=0)
            bpe_lens = torch.cat([bpe_lens, bpe_lens], dim=0)

        # Add char_lens by one for eos token
        abs_length = torch.round(bpe_lens * bpe.shape[1])

        # Append eos token at the end of the label sequences
        bpe_with_eos = append_eos_token(
            bpe, length=abs_length, eos_index=params.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / bpe_with_eos.shape[1]
        loss_seq = params.seq_cost(p_seq, bpe_with_eos, length=rel_length)

        if (
            stage == "train"
            and params.epoch_counter.current <= params.number_of_ctc_epochs
        ):
            loss_ctc = params.ctc_cost(p_ctc, bpe, wav_lens, bpe_lens)
            loss = (
                params.ctc_weight * loss_ctc
                + (1 - params.ctc_weight) * loss_seq
            )
        else:
            loss = loss_seq

        stats = {}
        if stage != "train":
            # Prediction
            word_seq = params.bpe_tokenizer(hyps, task="decode_from_list")
            char_seq = split_word(word_seq)
            # Truth
            words = undo_padding(words, word_lens)
            words = convert_index_to_lab(words, index2lab)
            chars = split_word(words)
            cer_stats = edit_distance.wer_details_for_batch(
                ids, chars, char_seq, compute_alignments=True
            )
            wer_stats = edit_distance.wer_details_for_batch(
                ids, words, word_seq, compute_alignments=True
            )
            # If needed, compute token error rate
            if params.ter_eval:
                bpe = undo_padding(bpe, bpe_lens)
                ter_stats = edit_distance.wer_details_for_batch(
                    ids, bpe, hyps, compute_alignments=True
                )
                stats["TER"] = ter_stats
            stats["CER"] = cer_stats
            stats["WER"] = wer_stats
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
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss, stats = self.compute_objectives(predictions, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        wer = summarize_error_rate(valid_stats["WER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, wer)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"WER": wer},
            importance_keys=[ckpt_recency, lambda c: -c.meta["WER"]],
        )

    def load_tokenizer(self):
        save_model_path = params.save_folder + "/tok_unigram.model"
        save_vocab_path = params.save_folder + "/tok_unigram.vocab"

        if hasattr(params, "tok_mdl_file"):
            download_file(
                params.tok_mdl_file, save_model_path, replace_existing=True
            )
            params.bpe_tokenizer.sp.load(save_model_path)
        if hasattr(params, "tok_voc_file"):
            download_file(
                params.tok_voc_file, save_vocab_path, replace_existing=True
            )

    def load_lm(self):
        save_model_path = params.output_folder + "/save/lm_model.ckpt"
        download_file(params.lm_ckpt_file, save_model_path)
        state_dict = torch.load(save_model_path)
        # Removing prefix
        state_dict = {k.split(".", 1)[1]: v for k, v in state_dict.items()}
        params.lm_model.load_state_dict(state_dict, strict=True)


# Prepare data
prepare_librispeech(
    data_folder=params.data_folder,
    splits=params.train_splits + [params.dev_split, params.test_split],
    merge_lst=params.train_splits,
    merge_name=params.csv_train,
    save_folder=params.data_folder,
)


train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(iter(train_set))

# if augmentation option is activate
# add it as a module and allow the .eval() mode
# to skip the perturbation during dev and test
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)

# Check if the model should be trained on multiple GPUs.
# Important: DataParallel MUST be called after the ASR (Brain) class init.
if params.multigpu:
    params.enc = torch.nn.DataParallel(params.enc)
    params.ctc_lin = torch.nn.DataParallel(params.ctc_lin)
    params.emb = torch.nn.DataParallel(params.emb)
    params.dec = torch.nn.DataParallel(params.dec)
    params.seq_lin = torch.nn.DataParallel(params.seq_lin)

# Load latest checkpoint to resume training
asr_brain.load_tokenizer()
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["WER"])
ids, words, word_lens = first_y
words = words.to(params.device)
words.fill_(params.bos_index)
# Only needs one timestep of input to initialize the weight
# Initialization has to be done before loading a heckpoint
if hasattr(params, "lm_ckpt_file"):
    beam_searcher.lm_forward_step(words[:, 0], memory=None)
    asr_brain.load_lm()

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
