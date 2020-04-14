import sys
import torch
import collections
import speechbrain.utils.edit_distance as edit_distance 
import speechbrain.data_io.wer as wer_io
from speechbrain.data_io.data_io import IterativeCSVWriter 
from speechbrain.decoders.ctc import ctc_greedy_decode
from tqdm.contrib import tzip
from speechbrain.core import Experiment
with open('recipes/ASR_CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml') as fi:
sb = Experiment(
    yaml_stream=fi,
    commandline_args=sys.argv[1:],
)


def main():

    # Prepare the data
    sb.copy_locally()
    sb.prepare_timit()
    train_set = sb.train_loader()
    valid_set = sb.valid_loader()
    test_set = sb.test_loader()

    # Load latest model
    sb.recover_if_possible()

    # training/validation epochs
    for epoch in sb.epoch_counter:

        # Iterate train and perform updates
        sb.model.train()
        train_losses = []
        for wav, phn in tzip(*train_set):
            ids, wav, wav_len = prepare_for_computations(wav)
            ids, phn, phn_len = prepare_for_computations(phn)
            pout = neural_computations(sb.model, wav, wav_len)
            detached_loss = learn(sb.model, pout, phn, wav_len, phn_len)
            train_losses.append(detached_loss)

        # Iterate validataion to check progress
        sb.model.eval()
        valid_losses = []
        valid_wer_stats = collections.Counter()
        for wav, phn in tzip(*valid_set):
            ids, wav, wav_len = prepare_for_computations(wav)
            ids, phn, phn_len = prepare_for_computations(phn)
            pout = neural_computations(sb.model, wav, wav_len)
            detached_loss, valid_wer_stats = validation(
                ids, pout, phn, wav_len, phn_len, valid_wer_stats)
            valid_losses.append(detached_loss)

        train_stats = {"loss": mean(train_losses)}
        valid_stats = {'loss': mean(valid_losses), 
                       'wer': valid_wer_stats['WER']}

        sb.lr_annealing([sb.optimizer], epoch, valid_stats['wer'])
        sb.save_and_keep_only({'wer': valid_stats['wer']}, min_keys=['wer'])
        sb.log_epoch_stats(epoch, train_stats, valid_stats)

    # Load best model, evaluate that: 
    sb.recover_if_possible(min_key='wer')
    sb.model.eval()
    details_by_utt = []
    with open(sb.predictions_file, "w") as fo:
        hyp_writer = IterativeCSVWriter(fo, ["predictions"])
        for wav, phn in tzip(*test_set):
            ids, wav, wav_len = prepare_for_computations(wav)
            ids, phn, phn_len = prepare_for_computations(phn)
            pout = neural_computations(sb.model, wav, wav_len)
            hyps, batch_details = evaluation(ids, pout, phn, wav_len, phn_len)
            details_by_utt.extend(batch_details)
            write_hyps(hyp_writer, hyps, wav_len, wav.shape[-1])  # Time last 

    summary_details = edit_distance.wer_summary(details_by_utt)
    wer_io.print_wer_summary(summary_details)
    wer_io.print_alignments(details_by_utt)


def prepare_for_computations(data):
    identifier, data, data_len = data
    return identifier, data.to(sb.device), data_len.to(sb.device)


def mean(loss):
    return float(sum(loss) / len(loss))


def to_output_format(ids, seqs):
    out = [[sb.train_loader.label_dict["phn"]["index2lab"][int(ind)]
            for ind in seq] for seq in seqs]
    return dict(zip(ids, out))


def neural_computations(model, wav, wav_len):
    feats = sb.compute_features(wav, wav_len)
    return model(feats)


def learn(model, pout, phn, wav_len, phn_len):
    loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
    loss.backward()
    sb.optimizer([model])
    return loss.detach()


def validation(ids, pout, phn, wav_len, phn_len, wer_stats):
    with torch.no_grad():
        loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
        batch_outputs = ctc_greedy_decode(pout, 
                wav_len, 
                blank_id = -1)
        wer_stats = edit_distance.accumulatable_wer_stats(
                phn.tolist(), batch_outputs, stats = wer_stats
        )
    return loss.detach(), wer_stats


def evaluation(ids, pout, phn, wav_len, phn_len):
    with torch.no_grad():
        batch_outputs = ctc_greedy_decode(pout, 
                wav_len,
                blank_id = -1)
        refs = to_output_format(ids, phn.tolist())
        hyps = to_output_format(ids, batch_outputs)
        details_by_utt = edit_distance.wer_details_by_utterance(refs, 
                hyps, 
                compute_alignments=True)
        return hyps, details_by_utt


def write_hyps(hyp_writer, hyps, wav_lens, max_len):
    for hyp_item, wav_len in zip(hyps.items(), wav_lens):
        duration = int(wav_len * max_len) / sb.sample_rate 
        ID, decoded = hyp_item
        hyp_writer.write(ID=ID, duration=duration, 
                predictions = f'"{" ".join(decoded)}"',
                predictions_format = "string")


if __name__ == '__main__':
    main()
