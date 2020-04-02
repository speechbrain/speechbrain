import sys
import torch
from tqdm import tqdm
from torch import autograd
from speechbrain.core import Experiment
sb = Experiment(
    yaml_stream=open('recipes/CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml'),
    commandline_args=sys.argv[1:],
)


def main():

    # Prepare the data
    sb.copy_locally()
    sb.prepare_timit()

    # training/validation epochs
    for epoch in range(sb.constants['N_epochs']):
        train_loss = {'loss': []}
        valid_loss = {'loss': [], 'wer': []}

        # Iterate train and perform updates
        for wav, phn in tqdm(zip(*sb.train_loader())):
            neural_computations(train_loss, sb.model, wav, phn, 'train')

        # Iterate validataion to check progress
        with torch.no_grad():
            for wav, phn in tqdm(zip(*sb.valid_loader())):
                neural_computations(valid_loss, sb.model, wav, phn, 'valid')

            sb.lr_annealing([sb.optimizer], epoch, mean(valid_loss['wer']))
            performance = {
                'loss_tr': mean(train_loss['loss']),
                'loss_valid': mean(valid_loss['loss']),
                'wer_valid': mean(valid_loss['wer']),
            }
            sb.save_checkpoint(epoch, performance)

    # Evaluate our model
    test_loss = {'loss': [], 'wer': []}
    for wav, phn in tqdm(zip(*sb.test_loader())):
        neural_computations(test_loss, sb.model, wav, phn, 'test')

    print("Final WER: %f" % mean(test_loss['wer']))


def mean(loss):
    return sum(loss) / len(loss)


def neural_computations(losses, model, wav, phn, mode):

    id, wav, wav_len = wav
    id, phn, phn_len = phn

    feats = sb.compute_features(wav.cuda())
    feats = sb.mean_var_norm(feats, wav_len.cuda())
    phn = phn.cuda()
    phn_len = phn_len.cuda()

    pout = model(feats)

    if mode == 'train':
        loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
        loss.backward()
        sb.optimizer([model])
        losses['loss'].append(loss.detach())
    else:
        loss, wer = sb.compute_cost_wer(pout, phn, [wav_len, phn_len])
        losses['loss'].append(loss.detach())
        losses['wer'].append(wer.detach())

    # if mode == 'test':
    #    print_predictions(id, pout, wav_len)


if __name__ == '__main__':
    main()
