import sys
import torch
import collections
import speechbrain.utils.edit_distance as edit_distance 
import speechbrain.data_io.wer as wer_io
from speechbrain.data_io.data_io import IterativeCSVWriter 
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding 
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

    # Initialize / Load latest model
    dummy_input = torch.rand([sb.batch_size, sb.n_mels, 100]).to(sb.device)
    sb.model.init_params(dummy_input)
    sb.optimizer.init_params([sb.model])
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
        with torch.no_grad():
            valid_losses = []
            valid_wer_details = []
            for wav, phn in tzip(*valid_set):
                ids, wav, wav_len = prepare_for_computations(wav)
                ids, phn, phn_len = prepare_for_computations(phn)
                pout = neural_computations(sb.model, wav, wav_len)
                detached_loss, wer_details = validation(
                    ids, pout, phn, wav_len, phn_len)
                valid_losses.append(detached_loss)
                valid_wer_details.extend(wer_details)
        valid_wer_summary = edit_distance.wer_summary(valid_wer_details)
        valid_wers = [utt['WER'] for utt in valid_wer_details]
        
        train_stats = {"loss": mean(train_losses)}
        valid_stats = {'loss': mean(valid_losses), 
                       'wer': valid_wer_summary['WER']}

        sb.lr_annealing([sb.optimizer], epoch, valid_wer_summary['WER'])
        sb.log_epoch_stats(epoch, train_stats, valid_stats)
        sb.save_and_keep_only({'wer': valid_wer_summary['WER']}, 
                min_keys=['wer'],
                end_of_epoch=True)

    # Load best model, evaluate that: 
    sb.recover_if_possible(min_key='wer')
    sb.model.eval()
    with torch.no_grad():
        details_by_utt = []
        ind2lab = sb.train_loader.label_dict["phn"]["index2lab"]
        with open(sb.predictions_file, "w") as fo:
            hyp_writer = IterativeCSVWriter(fo, ["predictions"])
            for wav, phn in tzip(*test_set):
                ids, wav, wav_len = prepare_for_computations(wav)
                ids, phn, phn_len = prepare_for_computations(phn)
                pout = neural_computations(sb.model, wav, wav_len)
                hyps, batch_details = evaluation(
                        ids, pout, phn, wav_len, phn_len, ind2lab)
                details_by_utt.extend(batch_details)
                write_hyps(hyp_writer, ids, 
                        hyps, wav_len, wav.shape[-1])  # Time last 

    summary_details = edit_distance.wer_summary(details_by_utt)
    with open(sb.wer_file, "w") as fo:
        wer_io.print_wer_summary(summary_details, file=fo)
        wer_io.print_alignments(details_by_utt, file=fo)
    # Print to stdout too:
    print("Final test set WER:")
    wer_io.print_wer_summary(summary_details)


def prepare_for_computations(data):
    identifier, data, data_len = data
    return identifier, data.to(sb.device), data_len.to(sb.device)


def mean(loss):
    return float(sum(loss) / len(loss))


def neural_computations(model, wav, wav_len):
    feats = sb.compute_features(wav)
    feats = sb.normalize(feats, wav_len)
    return model(feats)


def learn(model, pout, phn, wav_len, phn_len):
    loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
    loss.backward()
    sb.optimizer([model])
    return loss.detach()


def validation(ids, pout, phn, wav_len, phn_len):
    loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
    batch_outputs = ctc_greedy_decode(pout, 
            wav_len, 
            blank_id = -1)
    phn_unpadded = undo_padding(phn, phn_len)
    wer_details = edit_distance.wer_details_for_batch(
            ids, phn_unpadded, batch_outputs
    )
    return loss.detach(), wer_details


def evaluation(ids, pout, phn, wav_len, phn_len, ind2lab):
    batch_outputs = ctc_greedy_decode(pout, 
            wav_len,
            blank_id = -1)
    phn_unpadded = undo_padding(phn, phn_len)
    batch_outputs = convert_index_to_lab(
            batch_outputs, ind2lab)
    phn_unpadded = convert_index_to_lab(
            phn_unpadded, ind2lab)
    details_by_utt = edit_distance.wer_details_for_batch(
            ids,
            phn_unpadded, 
            batch_outputs, 
            compute_alignments=True)
    return batch_outputs, details_by_utt


def write_hyps(hyp_writer, ids, hyps, wav_lens, max_len):
    for ID, hyp_item, wav_len in zip(ids, hyps, wav_lens):
        duration = int(wav_len * max_len) / sb.sample_rate 
        hyp_writer.write(ID=ID, duration=duration, 
                predictions = f'"{" ".join(hyp_item)}"',
                predictions_format = "string")


if __name__ == '__main__':
    main()
