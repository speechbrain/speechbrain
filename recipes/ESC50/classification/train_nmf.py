import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from train_classifier import dataio_prep


class NMFBrain(sb.core.Brain):
    """
    The SpeechBrain class to train Non-Negative Factorization with Amortized Inference
    """

    def compute_forward(self, batch, stage=sb.Stage.TRAIN):
        """
        This function calculates the forward pass for NMF
        """

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        X_stft = self.hparams.compute_stft(wavs)
        X_stft_power = self.hparams.compute_stft_mag(X_stft)
        X_stft_tf = torch.log1p(X_stft_power)
        z = self.hparams.nmf_encoder(X_stft_tf)
        Xhat = torch.matmul(
            self.hparams.nmf_model.return_W("torch"), z.squeeze()
        )

        return Xhat

    def compute_objectives(self, target, predictions):
        loss = ((target.squeeze() - predictions) ** 2).mean()
        return loss


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    datasets, _ = dataio_prep(hparams)

    nmfbrain = NMFBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if not hparams["test_only"]:
        nmfbrain.fit(
            epoch_counter=nmfbrain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    test_stats = nmfbrain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # nmf_model = hparams["nmf"].to(hparams["device"])
    # nmf_encoder = hparams["nmf_encoder"].to(hparams["device"])
    # opt = torch.optim.Adam(
    #     lr=1e-4,
    #     params=list(nmf_encoder.parameters()) + list(nmf_model.parameters()),
    # )

    # for e in range(200):
    #     for i, element in enumerate(datasets["train"]):
    #         # print(element["sig"].shape[0] / hparams["sample_rate"])

    #         opt.zero_grad()
    #         Xs = hparams["compute_stft"](
    #             element["sig"].unsqueeze(0).to(hparams["device"])
    #         )
    #         Xs = hparams["compute_stft_mag"](Xs)
    #         Xs = torch.log(Xs + 1).permute(0, 2, 1)
    #         z = nmf_encoder(Xs)

    #         Xhat = torch.matmul(nmf_model.return_W("torch"), z.squeeze())
    #         loss = ((Xs.squeeze() - Xhat) ** 2).mean()
    #         loss.backward()

    #         opt.step()
    #         if 0:
    #             if i in [100]:
    #                 draw_fig()
    #     print("loss is {}, epoch is {} ".format(loss.item(), e))

    # torch.save(nmf_model.return_W("torch"), "nmf_decoder.pt")
