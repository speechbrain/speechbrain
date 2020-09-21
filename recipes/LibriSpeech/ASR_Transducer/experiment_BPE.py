#!/usr/bin/env python3
import os
import sys
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.transducer import (
    transducer_greedy_decode,
    transducer_beam_search_decode,
)
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import split_word
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate
import torch

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
    [params.enc, params.emb, params.dec, params.output]
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
        id, wavs, lens = x
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        if stage == "train":
            if hasattr(params, "env_corrupt"):
                wavs_noise = params.env_corrupt(wavs, lens, init_params)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

        if hasattr(params, "augmentation") and stage == "train":
            wavs = params.augmentation(wavs, lens, init_params)

        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, lens)
        # Transcription network: input-output dependency
        TN_output = params.enc(feats, init_params=init_params)
        if stage == "train":
            # Prediction network: output-output dependency
            # y contains a tuple of tensors (id, words, word_lens)
            ids, words, word_lens = y
            words = torch.cat([words, words], dim=0)
            word_lens = torch.cat([word_lens, word_lens])
            index2lab = params.train_loader.label_dict["wrd"]["index2lab"]
            bpe, _ = params.bpe_tokenizer(
                words,
                word_lens,
                index2lab,
                task="encode",
                init_params=init_params,
            )
            bpe = bpe.to(params.device)
            decoder_input = prepend_bos_token(bpe, bos_index=params.blank_index)
            PN_output = params.emb(decoder_input, init_params=init_params)
            PN_output, _ = params.dec(PN_output, init_params=init_params)
            # Joint the networks
            joint = params.Tjoint(
                TN_output.unsqueeze(2),
                PN_output.unsqueeze(1),
                init_params=init_params,
            )
            outputs = params.output(joint, init_params=init_params)
            outputs = params.log_softmax(outputs)
            return outputs, lens
        elif stage == "valid":
            hyps, scores = transducer_greedy_decode(
                TN_output,
                [params.emb, params.dec],
                params.Tjoint,
                [params.output],
                params.blank_index,
            )
            return hyps, scores
        else:
            if hasattr(params, "lm_model"):
                (
                    best_hyps,
                    best_scores,
                    nbest_hyps,
                    nbest_scores,
                ) = transducer_beam_search_decode(
                    TN_output,
                    [params.emb, params.dec],
                    params.Tjoint,
                    [params.output],
                    params.blank_index,
                    beam=params.beam,
                    nbest=params.nbest,
                    lm_module=params.lm_model,
                    lm_weight=params.lm_weight,
                )
            else:
                (
                    best_hyps,
                    best_scores,
                    nbest_hyps,
                    nbest_scores,
                ) = transducer_beam_search_decode(
                    TN_output,
                    [params.emb, params.dec],
                    params.Tjoint,
                    [params.output],
                    params.blank_index,
                    beam=params.beam,
                    nbest=params.nbest,
                )
            return best_hyps, best_scores

    def compute_objectives(self, predictions, targets, stage="train"):
        ids, words, word_lens = targets
        if stage != "train":
            if stage == "valid":
                index2lab = params.valid_loader.label_dict["wrd"]["index2lab"]
            else:
                index2lab = params.test_loader.label_dict["wrd"]["index2lab"]
            bpe, bpe_lens = params.bpe_tokenizer(
                words, word_lens, index2lab, task="encode"
            )
            bpe, bpe_lens = bpe.to(params.device), bpe_lens.to(params.device)
        else:
            predictions, lens = predictions
            index2lab = params.train_loader.label_dict["wrd"]["index2lab"]
            bpe, bpe_lens = params.bpe_tokenizer(
                words, word_lens, index2lab, task="encode"
            )
            bpe, bpe_lens = bpe.to(params.device), bpe_lens.to(params.device)
            if hasattr(params, "env_corrupt"):
                bpe = torch.cat([bpe, bpe], dim=0)
                bpe_lens = torch.cat([bpe_lens, bpe_lens], dim=0)
            loss = params.compute_cost(
                predictions,
                bpe.to(params.device).long(),
                lens.to(params.device),
                bpe_lens.to(params.device),
            )

        stats = {}
        if stage != "train":
            sequence_BPE, loss = predictions
            word_seq = params.bpe_tokenizer(
                sequence_BPE, task="decode_from_list"
            )
            char_seq = split_word(word_seq)

            words = undo_padding(words, word_lens)
            words = convert_index_to_lab(words, index2lab)
            chars = split_word(words)
            cer_stats = edit_distance.wer_details_for_batch(
                ids, chars, char_seq, compute_alignments=True
            )
            wer_stats = edit_distance.wer_details_for_batch(
                ids, words, word_seq, compute_alignments=True
            )
            if params.ter_eval:
                bpe = undo_padding(bpe, bpe_lens)
                ter_stats = edit_distance.wer_details_for_batch(
                    ids, bpe, sequence_BPE, compute_alignments=True
                )
                stats["TER"] = ter_stats
            stats["CER"] = cer_stats
            stats["WER"] = wer_stats

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        wer = summarize_error_rate(valid_stats["WER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, wer)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"WER": wer},
            importance_keys=[ckpt_recency, lambda c: -c.meta["WER"]],
            num_to_keep=10,
        )

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
        out = self.compute_forward(inputs, None, stage=stage)
        loss, stats = self.compute_objectives(out, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

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
    splits=["train-clean-100", "dev-clean", "test-clean"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(iter(train_set))

if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Create brain object for training
asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)

# Beamsearch with external LM
# Only needs one timestep of input to initialize the weight
# Initialization has to be done before loading a heckpoint
if hasattr(params, "lm_ckpt_file"):
    # fake an forward pass to initialisaze the model
    inp_tokens = torch.Tensor([[1, 2, 3]]).cuda()
    _, _ = params.lm_model(inp_tokens, init_params=True)
    asr_brain.load_lm()

# Check if the model should be trained on multiple GPUs.
# Important: DataParallel MUST be called after the ASR (Brain) class init.
if params.multigpu:
    params.enc = torch.nn.DataParallel(params.enc)
    params.emb = torch.nn.DataParallel(params.emb)
    params.dec = torch.nn.DataParallel(params.dec)
    params.output = torch.nn.DataParallel(params.output)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["WER"])
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
