#!/usr/bin/python
"""Recipe for training a speaker verification system based on contrastive
learning. We employ a pre-trained embedding extractors (e.g, xvectors)
followed by a binary discriminator trained with binary-cross-entropy.
The discriminator distinguishes between positive and negative examples
that are properly sampled from the dataset. Data augmentation is also
employed to significantly improved performance.

This approach is inspired from the following paper:

M. Ravanelli, Y. Bengio: "Learning Speaker Representations with Mutual
Information", Proc. of InterSpeech 2019
https://arxiv.org/abs/1812.00271

To run this recipe, do the following:
    > python speaker_verification_discriminator.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/verfication_discriminator_xvector.yaml

Author
* Mirco Ravanelli 2020
"""
import os
import sys
import torch
import logging
import torchaudio
import random
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from tqdm.contrib import tqdm
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file


# Trains (pre-trained) speaker embeddings + binary discriminator
class VerificationBrain(sb.core.Brain):
    def fit_batch(self, batch):
        out, target_discrim = self.compute_forward(batch)
        loss = self.compute_objectives(
            out, target_discrim, stage=sb.Stage.TRAIN
        )
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        out, target_discrim = self.compute_forward(batch)
        loss = self.compute_objectives(out, target_discrim, stage=stage)
        return loss.detach().cpu()

    def get_positive_sample(self, wav_anchor, seg_ids):
        """Samples other waveforms from the same speaker (positive samples)
        """
        wav_pos = torch.zeros_like(wav_anchor)
        for i, seg_id in enumerate(seg_ids):
            spk_id = seg_id.split("--")[0]

            if spk_id in wav_stored:
                wav_pos[i] = wav_stored[spk_id]
            else:
                wav_pos[i] = wav_anchor[i]

            wav_stored[spk_id] = wav_anchor[i].detach().clone()
        return wav_pos

    def get_negative_sample(self, wav_anchor):
        """Samples other waveforms from different speakers (negative samples)
        """
        wav_neg = torch.rand_like(wav_anchor) * 1e-08

        # Shuffle keys of the wav_stored dictionary
        rand_keys = list(wav_stored.keys())
        random.shuffle(rand_keys)
        rand_keys = rand_keys[0 : wav_anchor.shape[0]]

        # copying negative samples from the wav_stored dictionary
        for i, rand_key in enumerate(rand_keys):
            wav_neg[i] = wav_stored[rand_key]

        return wav_neg

    def gather_samples(self, emb):
        """Given anchor, positive and negative it returns all the samples
        and the targets needed to feed the binary discriminator. Positive
        samples concatenate anchor+positive embeddings, while negative samples
        concatenate anchor + negative ones.
        """

        # Managing augmented embeddings too
        emb_clean, emb_aug = emb.chunk(2)
        (emb_clean_anchor, emb_clean_pos, emb_clean_neg,) = emb_clean.chunk(3)
        emb_aug_anchor, emb_aug_pos, emb_aug_neg = emb_aug.chunk(3)

        # If I switch anchor + positive sample I have another positive
        # sample for free
        positive_clean = torch.cat([emb_clean_anchor, emb_clean_pos], dim=2)
        positive_clean2 = torch.cat([emb_clean_pos, emb_clean_anchor], dim=2)

        positive_noise = torch.cat([emb_aug_anchor, emb_aug_pos], dim=2)
        positive_noise2 = torch.cat([emb_aug_pos, emb_aug_anchor], dim=2)

        # Combining clean and noisy samples as well
        positive_mix = torch.cat([emb_clean_anchor, emb_aug_pos], dim=2)
        positive_mix2 = torch.cat([emb_clean_pos, emb_aug_anchor], dim=2)

        positive = torch.cat(
            [
                positive_clean,
                positive_noise,
                positive_mix,
                positive_clean2,
                positive_noise2,
                positive_mix2,
            ]
        )

        # Note: If I switch anchor + negative sample I have another negative
        # sample for free
        negative_clean = torch.cat([emb_clean_anchor, emb_clean_neg], dim=2)
        negative_clean2 = torch.cat([emb_clean_neg, emb_clean_anchor], dim=2)

        negative_noise = torch.cat([emb_aug_anchor, emb_aug_neg], dim=2)
        negative_noise2 = torch.cat([emb_aug_neg, emb_aug_anchor], dim=2)

        # Combining clean and noisy samples as well
        negative_mix = torch.cat([emb_clean_anchor, emb_aug_neg], dim=2)
        negative_mix2 = torch.cat([emb_clean_neg, emb_aug_anchor], dim=2)

        negative = torch.cat(
            [
                negative_clean,
                negative_noise,
                negative_mix,
                negative_clean2,
                negative_noise2,
                negative_mix2,
            ]
        )

        samples = torch.cat([positive, negative])

        targets = torch.cat(
            [
                torch.ones(positive.shape[0], device=positive.device),
                torch.zeros(positive.shape[0], device=positive.device),
            ]
        )
        targets = targets.unsqueeze(1).unsqueeze(1)
        return samples, targets

    def data_augmentation(self, wavs, lens):
        """Performs data augmentation given a batch of input waveforms.
        """
        # Environmental corruption + waveform augmentation
        wavs_aug = self.modules.env_corrupt(wavs, lens)
        wavs_aug = self.modules.augmentation(wavs_aug, lens)

        # Concatenate noisy and clean batches
        wavs = torch.cat([wavs, wavs_aug], dim=0)
        lens = torch.cat([lens, lens], dim=0)

        return wavs, lens

    def compute_embeddings(self, wavs, lens):
        """Computes the embeddings given a batch of input waveforms.
        """
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        if self.hparams.freeze_embeddings:
            self.modules.embedding_model.eval()
            with torch.no_grad():
                emb = self.modules.embedding_model(feats)
        else:
            emb = self.modules.embedding_model(feats)

        # Applying batch normaization
        emb = self.modules.bn_emb(emb)
        return emb

    def compute_embeddings_loop(self, data_loader):
        """Computes the embeddings of all the waveforms specified in the
        dataloader.
        """
        embedding_dict = {}

        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, dynamic_ncols=True):
                seg_ids = batch.id
                wavs, lens = batch.sig
                wavs, lens = (
                    wavs.to(self.device),
                    lens.to(self.device),
                )
                emb = self.compute_embeddings(wavs, lens)
                for i, seg_id in enumerate(seg_ids):
                    embedding_dict[seg_id] = emb[i].detach().clone()

        return embedding_dict

    def compute_forward(self, batch):
        """Computes the output of the speaker verification system composed of
        a (pre-trained) speaker embedding newtwork followed by a binary
        discriminator.
        """
        seg_ids = batch.id
        wav_anchor, lens = batch.sig
        wav_anchor = wav_anchor.to(self.device)
        lens = lens.to(self.device)

        # Get positive and negative samples
        wav_pos = self.get_positive_sample(wav_anchor, seg_ids)
        wav_neg = self.get_negative_sample(wav_anchor)
        wavs = torch.cat([wav_anchor, wav_pos, wav_neg])
        lens = torch.cat([lens, lens, lens])

        # Performing data augmentation and computing the embeddings
        wavs, lens = self.data_augmentation(wavs, lens)
        emb = self.compute_embeddings(wavs, lens)

        # Feeding positive and negative samples to the discriminator
        samples, target_discrim = self.gather_samples(emb)
        outputs = self.modules.discriminator(samples)

        return outputs, target_discrim

    def compute_objectives(self, outputs, target_discrim, stage):
        """Computes the Binary Cross-Entropy Loss (BPE) using targets derived
        from positive and negative samples.
        """
        loss = self.hparams.compute_cost(
            torch.nn.BCEWithLogitsLoss(reduction="none"),
            outputs,
            target_discrim,
            length=torch.ones(outputs.shape[0], device=outputs.device),
        )

        stats = {}
        if stage != sb.Stage.TRAIN:
            stats["loss"] = loss

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a epoch."""

        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            EER, min_dcf = self.verification_performance()
            stage_stats["ErrorRate"] = EER
            stage_stats["min_dcf"] = min_dcf

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

    def verification_performance(self,):
        """ Computes the EER using the standard voxceleb test split
        """
        # Computing  enrollment and test embeddings
        print("Computing enroll/test embeddings...")
        self.enrol_dict = self.compute_embeddings_loop(enrol_dataloader)
        self.test_dict = self.compute_embeddings_loop(test_dataloader)

        print("Computing EER..")
        # Reading standard verification split
        gt_file = os.path.join(
            self.hparams.data_folder, "meta", "veri_test.txt"
        )
        with open(gt_file) as f:
            veri_test = [line.rstrip() for line in f]

        positive_scores, negative_scores = self.get_verification_scores(
            veri_test
        )
        del self.enrol_dict, self.test_dict

        eer, th = EER(
            torch.tensor(positive_scores), torch.tensor(negative_scores)
        )

        min_dcf, th = minDCF(
            torch.tensor(positive_scores), torch.tensor(negative_scores)
        )
        return eer * 100, min_dcf

    def get_verification_scores(self, veri_test):
        """ computes positive and negative scores given the verification split.
        """
        samples = []
        labs = []
        positive_scores = []
        negative_scores = []
        self.modules.discriminator.eval()
        cnt = 0

        # Loop over all the verification tests
        for i, line in enumerate(veri_test):

            # Reading verification file (enrol_file test_file label)
            labs.append(int(line.split(" ")[0].rstrip().split(".")[0].strip()))
            enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
            sample = torch.cat(
                [self.enrol_dict[enrol_id], self.test_dict[test_id]], dim=1
            )
            samples.append(sample)

            # Gathering batches
            if cnt == self.hparams.batch_size - 1 or i == len(veri_test) - 1:
                samples = torch.cat(samples)
                with torch.no_grad():
                    outputs = self.modules.discriminator(samples)
                scores = torch.sigmoid(outputs)
                scores.detach()

                # Putting scores in the corresponding lists
                for j, score in enumerate(scores.tolist()):
                    if labs[j] == 1:
                        positive_scores.append(score[0])
                    else:
                        negative_scores.append(score[0])
                labs = []
                samples = []
                cnt = 0
                continue
            cnt = cnt + 1
        return positive_scores, negative_scores

    # Function for pre-trained model downloads
    def download_and_pretrain(self):
        """ Downloads the specified pre-trained model
        """
        save_model_path = (
            hparams["output_folder"] + "/save/embedding_model.ckpt"
        )
        if "http" in hparams["embedding_file"]:
            download_file(hparams["embedding_file"], save_model_path)
        hparams["embedding_model"].load_state_dict(
            torch.load(save_model_path), strict=True
        )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["enrol_annotation"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, enrol_data, test_data]
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # 4 Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **hparams["train_dataloader_opts"]
    )
    valid_dataloader = sb.dataio.dataloader.make_dataloader(
        valid_data, **hparams["valid_dataloader_opts"]
    )

    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **hparams["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **hparams["test_dataloader_opts"]
    )

    return train_dataloader, valid_dataloader, enrol_dataloader, test_dataloader


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    # This flag enable the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "splits": ["train", "dev", "test"],
            "split_ratio": [90, 10],
            "seg_dur": int(hparams["sentence_len"]) * 100,
        },
    )

    (
        train_dataloader,
        valid_dataloader,
        enrol_dataloader,
        test_dataloader,
    ) = dataio_prep(hparams)

    # Dictionary to store the last waveform read for each speaker
    wav_stored = {}

    # Brain class initialization
    verifier = VerificationBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Pre-train embeddings
    if hparams["pretrain_embeddings"]:
        verifier.download_and_pretrain()

    # Train the speaker verification model
    verifier.fit(
        hparams["epoch_counter"],
        train_set=train_dataloader,
        valid_set=valid_dataloader,
    )

    logger.info("Speaker verification model training completed!")
