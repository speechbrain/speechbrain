import torch
from speechbrain.core import load_params
f, params = load_params(__file__, 'params.yaml')


def neural_computations(wav, phn, mode):

    id, wav, wav_len = wav
    id, phn, phn_len = phn

    feats = f.compute_features([wav])
    feats = f.mean_var_norm([feats, wav_len])

    out = f.RNN([feats])
    out = f.lin([out])

    pout = f.softmax([out])

    if mode == 'train':
        loss = f.compute_cost([pout, phn, [wav_len, phn_len]])
        loss.backward()
        f.optimizer([f.RNN, f.lin])
        return loss
    else:
        return f.compute_cost_wer([pout, phn, [wav_len, phn_len]])

    # if mode == 'test':
    #    print_predictions(id, pout, wav_len)


f.copy_locally([])
f.prepare_timit([])

# training/validation epochs
train_data = f.train_loader([])
valid_data = f.valid_loader([])
test_data = f.test_loader([])

for epoch in range(int(params['N_epochs'])):
    tr_loss = []
    for wav, phn in zip(*train_data[0]):
        tr_loss.append(neural_computations(wav, phn, 'train').detach())

    valid_loss, valid_wer = [], []
    for wav, phn in zip(*valid_data[0]):
        loss, wer = neural_computations(wav, phn, 'valid')
        valid_loss.append(loss.detach())
        valid_wer.append(wer.detach())

    f.lr_annealing([epoch, torch.mean(wer)])
    performance = {
        'loss_tr': sum(tr_loss) / len(tr_loss),
        'loss_valid': sum(valid_loss) / len(valid_loss),
        'wer_valid': sum(valid_wer) / len(valid_wer),
    }
    f.save_checkpoint([epoch, performance])

# test
losses, wers = [], []
for wav, phn in zip(*test_data[0]):
    loss, wer = neural_computations(wav, phn, 'test')
    losses.append(loss)
    wers.append(wer)

print("Final WER: %f" % (sum(wers) / len(wers)))
