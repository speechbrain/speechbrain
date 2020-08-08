import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.nnet.losses import weighted_mse_loss
from tqdm.contrib import tqdm
from speechbrain.utils.data_utils import download_file
import uis_rnn_utils
import numpy as np
from torch import nn


# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)
# from voxceleb_prepare import prepare_voxceleb  # noqa E402

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    # params_to_save=params_file,
    overrides=overrides,
)

rnn_model = torch.nn.ModuleList(
    [
        params.rnn_model,
        params.first_linear_layer,
        params.activation,
        params.output_linear_layer,
    ]
)


# Definition of the steps for xvector computation from the waveforms
def preprocessing(wavs, lens, init_params=False):
    with torch.no_grad():
        wavs = wavs.to(params.device)
        feats = params.compute_features(wavs, init_params=init_params)
        feats = params.mean_var_norm(feats, lens)
        x_vect = params.xvector_model(feats, init_params=init_params)
        if init_params:
            if "https://" in params.xvector_file:
                download_and_pretrain()
            else:
                params.xvector_model.load_state_dict(
                    torch.load(params.xvector_file), strict=True
                )
    return x_vect


# Function for pre-trained model downloads
def download_and_pretrain():
    save_model_path = params.output_folder + "/save/xvect.ckpt"
    download_file(params.xvector_file, save_model_path)
    params.xvector_model.load_state_dict(
        torch.load(save_model_path), strict=True
    )


logger.info("Extracting xvectors...")

tr_xvectors = []
tr_ids = []


class SpeakerDiarization(sb.core.Brain):
    def __init__(self, modules=None, optimizer=None, first_inputs=None):
        self.modules = torch.nn.ModuleList(modules)
        self.rnn_init_hidden = torch.nn.Parameter(
            torch.zeros(params.rnn_layers, 1, params.rnn_hidden_size).to(
                params.device
            )
        )
        self.optimizer = optimizer
        self.avg_train_loss = 0.0
        self.sigma2 = 0.1
        self.sigma2 = torch.nn.Parameter(
            self.sigma2 * torch.ones(params.output_size).to(params.device)
        )
        self.transition_bias = None  # for prediction
        self.transition_bias_denominator = 0.0  # for prediction
        self.crp_alpha = 1.0
        self.sigma_alpha = 1.0
        self.sigma_beta = 1.0

        if first_inputs is not None:
            self.preprocess(first_inputs, init_params=True)

            if self.optimizer is not None:
                self.optimizer.init_params(self.modules)
                self.optimizer.optim.add_param_group(
                    {"params": self.rnn_init_hidden}
                )
                self.optimizer.optim.add_param_group({"params": self.sigma2})

    def preprocess(self, x, init_params=False):
        with torch.no_grad():
            ids, wavs, lens = x
            wavs = wavs.to(params.device)
            feats = params.compute_features(wavs, init_params=init_params)
            feats = params.mean_var_norm(feats, lens)
            x_vect = params.xvector_model(feats, init_params=init_params)[
                :, :, :256
            ]
            if init_params:
                if "https://" in params.xvector_file:
                    download_and_pretrain()
                else:
                    params.xvector_model.load_state_dict(
                        torch.load(params.xvector_file), strict=True
                    )
                self.compute_forward(x_vect, init_params=True)
        return x_vect

    def compute_forward(self, x, stage="train", init_params=False):
        h_state = self.rnn_init_hidden.repeat(1, params.batch_size, 1).to(
            params.device
        )

        rnn_out, _ = params.rnn_model(x, h_state, init_params=init_params)
        if isinstance(rnn_out, torch.nn.utils.rnn.PackedSequence):
            rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_out, batch_first=False
            )
        l1_out = params.first_linear_layer(rnn_out, init_params=init_params)
        l2_out = params.output_linear_layer(
            params.activation(l1_out), init_params=init_params
        )
        mean = torch.cumsum(l2_out, dim=2)
        mean_size = mean.size()
        mean = torch.mm(
            torch.diag(
                1.0
                / torch.arange(1, mean_size[0] + 1).float().to(params.device)
            ),
            mean.view(mean_size[0], -1),
        )
        mean = mean.view(mean_size)
        return mean

    def compute_objectives(self, predictions, targets, stage="train"):
        loss1 = weighted_mse_loss(
            (targets != 0).float() * predictions[:-1, :, :],
            targets,
            1 / (2 * self.sigma2),
        )

        # Sigma2 prior part.
        weight = (
            ((targets != 0).float() * predictions[:-1, :, :] - targets) ** 2
        ).view(-1, params.output_size)
        num_non_zero = torch.sum((weight != 0).float(), dim=0).squeeze()

        loss2 = (
            (2 * self.sigma_alpha + num_non_zero + 2)
            / (2 * num_non_zero)
            * torch.log(self.sigma2)
        ).sum() + (self.sigma_beta / (self.sigma2 * num_non_zero)).sum()

        # Regularization part.

        loss = loss1 + loss2
        return (
            loss,
            {
                "training_loss": loss,
                "weighted_mse_loss": loss1.data,
                "sigma2_prior": loss2.data,
            },
        )

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.modules.parameters(), params.grad_max_norm
        )
        self.optimizer.step()
        self.sigma2.data.clamp_(min=1e-6)
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def fit(
        self,
        train_iterations_counter,
        train_set,
        valid_set=None,
        progressbar=True,
    ):
        disable = not progressbar
        all_xvectors = torch.Tensor().to(params.device)
        all_spkid = []
        with tqdm(train_set, dynamic_ncols=True, disable=disable) as t:
            for i, batch in enumerate(t):
                wav, target = batch[0], batch[1][0]
                all_xvectors = torch.cat([all_xvectors, self.preprocess(wav)])
                all_spkid.extend(target)
        all_spkid = np.array(["_".join(i.split("_")[:-1]) for i in all_spkid])

        all_xvectors = all_xvectors.squeeze(1)

        sub_sequences, seq_lengths = uis_rnn_utils.resize_sequence(
            all_xvectors, all_spkid, num_permutations=10
        )

        with tqdm(
            train_iterations_counter, dynamic_ncols=True, disable=disable
        ) as t:
            for iteration in range(train_iterations_counter):
                self.modules.train()
                train_stats = {}
                disable = not progressbar
                packed_train_sequence, rnn_truth = uis_rnn_utils.pack_sequence(
                    sub_sequences,
                    seq_lengths,
                    params.batch_size,
                    params.output_size,
                    params.device,
                )
                batch = [packed_train_sequence, rnn_truth]
                stats = self.fit_batch(batch)
                self.add_stats(train_stats, stats)
                average = self.update_average(stats, iteration=i + 1)
                t.set_postfix(train_loss=average)

            valid_stats = {}
            if valid_set is not None:
                self.modules.eval()
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=disable
                    ):
                        stats = self.evaluate_batch(batch, stage="valid")
                        self.add_stats(valid_stats, stats)

            self.on_epoch_end(iteration, train_stats, valid_stats)


train_set = params.train_loader()
first_x, first_y = next(iter(train_set))
speaker_diarization_brain = SpeakerDiarization(
    modules=rnn_model,
    optimizer=params.optimizer,
    first_inputs=[
        first_x[0][: params.batch_size],
        first_x[1][: params.batch_size],
        first_x[2][: params.batch_size],
    ],
)


# checkpointer.recover_if_possible()
speaker_diarization_brain.fit(params.train_iterations, train_set)
