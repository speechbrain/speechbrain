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
    hyperparams/verfication_discriminator_xvector_voxceleb1.yaml

Author
* Mirco Ravanelli 2020
"""
import os
import sys
import torch
import random
import speechbrain as sb
from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER
from speechbrain.utils.data_utils import download_file


# Trains (pre-trained) speaker embeddings + binary discriminator
class VerificationBrain(sb.core.Brain):
    def fit_batch(self, batch):
        inputs, _ = batch
        out, target_discrim = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(out, target_discrim)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="test"):
        inputs, target_class = batch
        out, target_discrim = self.compute_forward(inputs, stage=stage)
        loss, stats = self.compute_objectives(out, target_discrim, stage=stage)
        stats["loss"] = loss.detach()
        return stats

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

    def data_augmentation(self, wavs, lens, init_params=False):
        """Performs data augmentation given a batch of input waveforms.
        """
        # Environmental corruption + waveform augmentation
        wavs_aug = params.env_corrupt(wavs, lens, init_params)
        wavs_aug = params.augmentation(wavs_aug, lens, init_params)

        # Concatenate noisy and clean batches
        wavs = torch.cat([wavs, wavs_aug], dim=0)
        lens = torch.cat([lens, lens], dim=0)

        return wavs, lens

    def compute_embeddings(self, wavs, lens, init_params=False):
        """Computes the embeddings given a batch of input waveforms.
        """
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        if params.freeze_embeddings:
            params.embedding_model.eval()
            with torch.no_grad():
                emb = params.embedding_model(feats, init_params=init_params)
        else:
            emb = params.embedding_model(feats, init_params=init_params)

        # Applying batch normaization
        emb = params.bn_emb(emb, init_params=init_params)
        return emb

    def compute_embeddings_loop(self, data_loader):
        """Computes the embeddings of all the waveforms specified in the
        dataloader.
        """
        embedding_dict = {}

        self.modules.eval()
        with torch.no_grad():
            for (batch,) in tqdm(data_loader, dynamic_ncols=True):
                seg_ids, wavs, lens = batch
                wavs, lens = wavs.to(params.device), lens.to(params.device)
                emb = self.compute_embeddings(wavs, lens, init_params=False)
                for i, seg_id in enumerate(seg_ids):
                    embedding_dict[seg_id] = emb[i].detach().clone()
        return embedding_dict

    def compute_forward(self, x, stage="train", init_params=False):
        """Computes the output of the speaker verification system composed of
        a (pre-trained) speaker embedding newtwork followed by a binary
        discriminator.
        """
        seg_ids, wav_anchor, lens = x

        # Get positive and negative samples
        wav_anchor, lens = wav_anchor.to(params.device), lens.to(params.device)
        wav_pos = self.get_positive_sample(wav_anchor, seg_ids)
        wav_neg = self.get_negative_sample(wav_anchor)
        wavs = torch.cat([wav_anchor, wav_pos, wav_neg])
        lens = torch.cat([lens, lens, lens])

        # Performing data augmentation and computing the embeddings
        wavs, lens = self.data_augmentation(wavs, lens, init_params)
        emb = self.compute_embeddings(wavs, lens, init_params)

        # Feeding positive and negative samples to the discriminator
        samples, target_discrim = self.gather_samples(emb)
        outputs = params.discriminator(samples, init_params=init_params)

        return outputs, target_discrim

    def compute_objectives(self, outputs, target_discrim, stage="train"):
        """Computes the Binary Cross-Entropy Loss (BPE) using targets derived
        from positive and negative samples.
        """
        loss = params.compute_cost(
            torch.nn.BCEWithLogitsLoss(reduction="none"),
            outputs,
            target_discrim,
            length=torch.ones(outputs.shape[0], device=outputs.device),
        )

        stats = {}
        if stage != "train":
            stats["loss"] = loss

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        EER = self.compute_EER()
        valid_stats["EER"] = [EER]

        old_lr, new_lr = params.lr_annealing(
            [params.optimizer], epoch, valid_stats["loss"]
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        params.checkpointer.save_and_keep_only(
            meta={"EER": EER}, min_keys=["EER"]
        )

    def compute_EER(self,):
        """ Computes the EER using the standard voxceleb test split
        """
        # Computing  enrollment and test embeddings
        print("Computing enroll/test embeddings...")
        self.enrol_dict = self.compute_embeddings_loop(enrol_set_loader)
        self.test_dict = self.compute_embeddings_loop(test_set_loader)

        print("Computing EER..")
        # Reading standard verification split
        gt_file = os.path.join(params.data_folder, "meta", "veri_test.txt")
        with open(gt_file) as f:
            veri_test = [line.rstrip() for line in f]

        positive_scores, negative_scores = self.get_verification_scores(
            veri_test
        )
        del self.enrol_dict, self.test_dict

        eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        return eer * 100

    def get_verification_scores(self, veri_test):
        """ computes positive and negative scores given the verification split.
        """
        samples = []
        labs = []
        positive_scores = []
        negative_scores = []
        params.discriminator.eval()
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
            if cnt == params.batch_size - 1 or i == len(veri_test) - 1:
                samples = torch.cat(samples)
                with torch.no_grad():
                    outputs = params.discriminator(samples)
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
        save_model_path = params.output_folder + "/save/embedding_model.ckpt"
        if "http" in params.embedding_file:
            download_file(params.embedding_file, save_model_path)
        params.embedding_model.load_state_dict(
            torch.load(save_model_path), strict=True
        )


# Begin Recipe!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from voxceleb_prepare import prepare_voxceleb  # noqa E402

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = sb.yaml.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=params.data_folder,
        save_folder=params.save_folder,
        splits=["train", "dev", "test"],
        split_ratio=[90, 10],
        seg_dur=300,
        rand_seed=params.seed,
    )

    # Dictionary to store the last waveform read for each speaker
    wav_stored = {}

    # Data loaders
    train_set = params.train_loader()
    valid_set = params.valid_loader()
    enrol_set_loader = params.enrol_loader()
    test_set_loader = params.test_loader()

    # Speaker verification Model
    modules = [
        params.embedding_model,
        params.discriminator,
        params.bn_emb,
    ]
    first_x, first_y = next(iter(train_set))

    # Object initialization for training the speaker verification model
    verifier = VerificationBrain(
        modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
    )

    # Pre-train embeddings
    if params.pretrain_embeddings:
        verifier.download_and_pretrain()

    # Recover checkpoints
    params.checkpointer.recover_if_possible()

    # Train the speaker verification model
    verifier.fit(
        params.epoch_counter, train_set=train_set, valid_set=valid_set,
    )

    print("Speaker verification model training completed!")
