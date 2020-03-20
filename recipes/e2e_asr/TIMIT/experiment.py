import torch
from speechbrain.core import load_params
sb, params = load_params('params.yaml')


def neural_computations(wav, phn, mode):

    id, wav, wav_len = wav
    id, phn, phn_len = phn

    feats = sb.compute_features(wav)
    feats = sb.mean_var_norm(feats, wav_len)

    out = sb.RNN(feats)
    out = sb.lin(out)

    pout = sb.softmax(out)

    if mode == 'train':
        loss = sb.compute_cost(pout, phn, [wav_len, phn_len])
        loss.backward()
        sb.optimizer([sb.RNN, sb.lin])
        return loss
    else:
        return sb.compute_cost_wer(pout, phn, [wav_len, phn_len])

    # if mode == 'test':
    #    print_predictions(id, pout, wav_len)


# Prepare the data
sb.copy_locally()
sb.prepare_timit()

# training/validation epochs
for epoch in range(int(params['N_epochs'])):
    tr_loss = []
    for wav, phn in zip(*sb.train_loader()):
        tr_loss.append(neural_computations(wav, phn, 'train').detach())
        break

    with torch.no_grad():
        valid_loss, valid_wer = [], []
        for wav, phn in zip(*sb.valid_loader()):
            loss, wer = neural_computations(wav, phn, 'valid')
            valid_loss.append(loss.detach())
            valid_wer.append(wer.detach())
            break

        sb.lr_annealing([sb.optimizer], epoch, torch.mean(wer))
        performance = {
            'loss_tr': sum(tr_loss) / len(tr_loss),
            'loss_valid': sum(valid_loss) / len(valid_loss),
            'wer_valid': sum(valid_wer) / len(valid_wer),
        }
        sb.save_checkpoint(epoch, performance)
    break

# test
losses, wers = [], []
for wav, phn in zip(*sb.test_loader()):
    loss, wer = neural_computations(wav, phn, 'test')
    losses.append(loss)
    wers.append(wer)
    break

print("Final WER: %f" % (sum(wers) / len(wers)))
