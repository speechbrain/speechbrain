import sys
import torch
from tqdm import tqdm
from torch import autograd
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
        for wav, phn in tqdm(zip(*train_set), total=len(train_set[0])):
            neural_computations(train_loss, sb.model, wav, phn, 'train')
        train_loss['loss'] = float(mean(train_loss['loss']))

        # Iterate validataion to check progress
        valid_loss = {'loss': [], 'wer': []}
        for wav, phn in tqdm(zip(*valid_set), total=len(valid_set[0])):
            neural_computations(valid_loss, sb.model, wav, phn, 'valid')

        valid_loss = {key: float(mean(valid_loss[key])) for key in valid_loss}
        sb.lr_annealing([sb.optimizer], epoch, valid_loss['wer'])
        sb.save_and_keep_only({'wer': valid_loss['wer']}, min_keys=['wer'])
        sb.log_epoch_stats(epoch, train_loss, valid_loss)

    # Evaluate our model
    test_loss = {'loss': [], 'wer': []}
    sb.recover_if_possible(min_key='wer')
    for wav, phn in tqdm(zip(*test_set), total=len(test_set[0])):
        neural_computations(test_loss, sb.model, wav, phn, 'test')

    print("Final WER: %f" % mean(test_loss['wer']))


def mean(loss):
    return sum(loss) / len(loss)


def neural_computations(losses, model, wav, phn, mode):

    id, wav, wav_len = wav
    id, phn, phn_len = phn

    feats = sb.compute_features(wav.cuda(), wav_len.cuda())

    if mode == 'train':
        model.train()
        pout = model(feats)
        loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
        loss.backward()
        sb.optimizer([model])
        losses['loss'].append(loss.detach())
    else:
        with torch.no_grad():
            model.eval()
            pout = model(feats)
            loss, wer = sb.compute_cost_wer(pout, phn, [wav_len, phn_len])
            losses['loss'].append(loss.detach())
            losses['wer'].append(wer.detach())


if __name__ == '__main__':
    main()
