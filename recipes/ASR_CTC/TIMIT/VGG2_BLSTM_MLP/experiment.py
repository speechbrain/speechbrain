import sys
import torch
import collections
import speechbrain.utils.edit_distance as edit_distance 
import speechbrain.data_io.wer as wer_io
from speechbrain.data_io.data_io import filter_ctc_output
from tqdm.contrib import tzip
from speechbrain.core import Experiment
sb = Experiment(
    yaml_stream=open('recipes/ASR_CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml'),
    commandline_args=sys.argv[1:],
)


def main():

    # Prepare the data
    sb.copy_locally()
    sb.prepare_timit()
    train_set = sb.train_loader()
    valid_set = sb.valid_loader()
    test_set = sb.test_loader()

    # Load best (lowest) WER model
    sb.recover_if_possible(min_key='wer')

    # training/validation epochs
    for epoch in sb.epoch_counter:

        # Iterate train and perform updates
        train_loss = {'loss': []}
        for wav, phn in tzip(*train_set):
            neural_computations(train_loss, sb.model, wav, phn, 'train')
        train_loss['loss'] = mean(train_loss['loss'])

        # Iterate validataion to check progress
        valid_loss = {'loss': [], 'wer_stats': collections.Counter()}
        for wav, phn in tzip(*valid_set):
            neural_computations(valid_loss, sb.model, wav, phn, 'valid')
        valid_stats = {'loss': mean(valid_loss['loss']), 
                       'wer': valid_loss['wer_stats']['WER']}

        sb.lr_annealing([sb.optimizer], epoch, valid_stats['wer'])
        sb.save_and_keep_only({'wer': valid_stats['wer']}, min_keys=['wer'])
        sb.log_epoch_stats(epoch, train_loss, valid_stats)

    # Evaluate our model
    test_loss = {'wer_details': []}
    sb.recover_if_possible(min_key='wer')
    for wav, phn in tzip(*test_set):
        neural_computations(test_loss, sb.model, wav, phn, 'test')

    summary_details = edit_distance.wer_summary(test_loss['wer_details'])
    wer_io.print_wer_summary(summary_details)
    wer_io.print_alignments(test_loss['wer_details'])


def mean(loss):
    return float(sum(loss) / len(loss))


def ctc_greedy_decode(probabilities, seq_lens, blank_id):
    batch_max_len = probabilities.shape[-1]  # Time last
    batch_outputs = []
    for seq, seq_len in zip(probabilities, seq_lens):
        actual_size = int(
            torch.round(seq_len * batch_max_len)
        )
        scores, predictions = torch.max(seq.narrow(-1, 0, actual_size), dim=0)
        out = filter_ctc_output(
                predictions.tolist(), blank_id=blank_id
        )
        batch_outputs.append(out)
    return batch_outputs


def to_output_format(ids, seqs):
    out = [[sb.train_loader.label_dict["phn"]["index2lab"][int(ind)]
            for ind in seq] for seq in seqs]
    return dict(zip(ids, out))


def neural_computations(losses, model, wav, phn, mode, ):

    id, wav, wav_len = wav
    id, phn, phn_len = phn
    wav, wav_len = wav.cuda(), wav_len.cuda()
    phn, phn_len = phn.cuda(), phn_len.cuda()

    feats = sb.compute_features(wav, wav_len)

    if mode == 'train':
        model.train()
        pout = model(feats)
        loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
        loss.backward()
        sb.optimizer([model])
        losses['loss'].append(loss.detach())
    elif mode == 'valid':
        with torch.no_grad():
            model.eval()
            pout = model(feats)
            loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
            losses['loss'].append(loss.detach())
            batch_outputs = ctc_greedy_decode(pout, 
                    wav_len, 
                    blank_id = sb.compute_cost.blank_index)
            losses['wer_stats'] = edit_distance.accumulatable_wer_stats(
                    phn.tolist(), batch_outputs, stats = losses['wer_stats']
            )
    elif mode == 'test':
        with torch.no_grad():
            model.eval()
            pout = model(feats)
            batch_outputs = ctc_greedy_decode(pout, 
                    wav_len,
                    blank_id = sb.compute_cost.blank_index)
            refs = to_output_format(id, phn.tolist())
            hyps = to_output_format(id, batch_outputs)
            details_by_utt = edit_distance.wer_details_by_utterance(refs, 
                    hyps, 
                    compute_alignments=True)
            losses['wer_details'].extend(details_by_utt)


if __name__ == '__main__':
    main()
