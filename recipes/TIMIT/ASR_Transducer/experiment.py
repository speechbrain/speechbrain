#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.transducer import decode_batch
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import (
    FileTrainLogger,
    summarize_average,
    summarize_error_rate,
)

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import TIMITPreparer  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

train_logger = FileTrainLogger(
    save_file=params.train_log,
    summary_fns={"loss": summarize_average, "PER": summarize_error_rate},
)
checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model_TN": params.model_TN,
        "model_PN": params.model_PN,
        "model_OUTN": params.model_OUTN,
        "optimizer": params.optimizer,
        "scheduler": params.lr_annealing,
        "normalizer": params.normalize,
        "counter": params.epoch_counter,
    },
)


# Define training procedure
class ASR(sb.core.Brain):
    def forward(
        self, 
        *input_args,
        init_params=True):
        if len(input_args)==2:
            ids, wavs, wav_lens = input_args[0]
            wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
            ids, target_seq, target_len = input_args[1]
            target_seq, target_len = target_seq.to(params.device), target_len.to(params.device)
            train=True    
        else:
            ids, wavs, wav_lens = input_args[0]
            wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
            train=False
            
        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        # Transcription Network: input-output dependency
        TN_output= params.model_TN(feats, init_params)
        if train:
            # Generate input seq for Prediction Network
            # y=[y_0,y_1,...y_n-1] => y=[blank,y_0,y_1,....,y_n-1]
            blank_vect=torch.ones((target_seq.size(0),1),device=params.device,dtype=torch.int64)*params.blank_id
            prediction_seq=torch.cat((blank_vect,target_seq.long()),dim=1)
            # Prediction Network: output-output dependency
            PN_output= params.embedding_PN(prediction_seq,init_params)
            PN_output, _ = params.model_PN(PN_output, init_params)
            PN_output = params.model_PN_lin(PN_output, init_params)
            #PN_output= params.model_PN_lin(PN_output, init_params)
            # Joint networks
            joint = params.Tjoint(TN_output.unsqueeze(2),PN_output.unsqueeze(1))
            # Output network
            transducer_output = params.model_OUTN(joint,init_params)
            transducer_output = params.head(transducer_output,init_params)
        else:
            transducer_output = decode_batch(TN_output,
                                    [params.embedding_PN,params.model_PN,params.model_PN_lin],
                                    params.Tjoint,
                                    [params.model_OUTN,params.head],params.blank_id)
            
            transducer_output = params.log_softmax(transducer_output)

        return transducer_output, wav_lens

    def compute_objectives(self, predictions, targets, train=True):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        loss = params.compute_cost(pout, phns, [pout_lens, phn_lens])
        if not train:
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"PER": stats}
            return loss, stats

        return loss

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )


#prepare = TIMITPreparer(
#    data_folder=params.data_folder,
#    splits=["train", "dev", "test"],
#    save_folder=params.data_folder,
#)
#prepare()
train_set = params.train_loader()
valid_set = params.valid_loader()
modules = [params.model_TN,params.model_PN, params.model_PN_lin, params.model_OUTN, params.head]
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_input=[next(iter(train_set[0])),next(iter(train_set[1]))],
    seq2seq=True,
)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
test_stats = asr_brain.evaluate(params.test_loader())
train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
