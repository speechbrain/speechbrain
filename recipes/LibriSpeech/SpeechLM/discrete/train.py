#!/usr/bin/env/python3
"""
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HOME="/scratch/adelmou/hf_home/" HF_HUB_CACHE="/scratch/adelmou/hf_home/hub/" python train.py hparams/ssl.yaml --data_folder=$SLURM_TMPDIR/LibriSpeech/
"""

import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from speechbrain.lobes.models.huggingface_transformers.discrete_speechlm import (
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
    InterleavedCodebookPattern,
)
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        tokens, length = batch.tokens
        print(tokens)
        print(length)
        #TODO: modify in/out so that we have the target
        # Forward pass
        # Feature extraction and attention pooling
        exit()
        with torch.no_grad():
            self.hparams.codec.to(self.device).eval()
            tokens, _, _ = self.hparams.codec(
                wavs, 
                wav_lens, 
                self.hparams.ssl_layer_num,
                self.hparams.deduplicate,
                self.hparams.bpe_tokenizer_path,
                # , n_quantizers=self.hparams.num_codebooks
            )
        # concat eos_token to the end of the sequence
        eos_token = torch.full((tokens.size(0), 1, 6), 1001, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, eos_token], dim=1)
        # permute dim 1 with dim 2
        tokens = tokens.permute(0, 2, 1)
        tokens = tokens[:, :self.hparams.config.n_codebooks, :self.hparams.config.block_size + 1]
        inp_tokens = tokens[:, :, :-1]
        target_tokens = tokens[:, :, 1:]
        patterned_inp = self.hparams.codebook_pattern.apply_delay_pattern(inp_tokens)
        # print(patterned_inp.shape)
        logits = self.modules.model(patterned_inp)
        undelayed_logits, undelayed_logits_mask = self.hparams.codebook_pattern.undelay_logits(logits)
        return undelayed_logits, undelayed_logits_mask, target_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        # unpack batch
        logits, logits_mask, target_tokens = predictions
        logits = logits.float()
        # calculate the loss on coarse audio tokens
        lm_coarse_loss = F.cross_entropy(
            logits[:, 0][logits_mask[:, 0]],
            target_tokens[:, 0][logits_mask[:, 0]],
            ignore_index=-1, 
            reduction='sum',
        )
        coarse_loss = lm_coarse_loss / (logits_mask[:, 0]).sum()
        # lm_loss = 0
        # if self.hparams.num_codebooks > 1:
        #     # calculate the loss on fine audio tokens
        #     lm_fine_loss = F.cross_entropy(logits[:, 1:][logits_mask[:, 1:]],
        #                                     target_tokens[:, 1:][logits_mask[:, 1:]], 
        #                                     ignore_index=-1, reduction='sum')
        #     fine = lm_fine_loss / (logits_mask[:, 1:]).sum()
        #     lm_loss += ((lm_coarse_loss + self.hparams.fine_weight * lm_fine_loss)) / logits_mask.sum()
        # else:
        #     fine = 0.
        #     lm_loss += coarse_loss / (logits_mask[:, 1:]).sum()

        return coarse_loss
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 1. Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]
    
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens
        
    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # # 2. Define audio pipeline:
    # @sb.utils.data_pipeline.takes("wav")
    # @sb.utils.data_pipeline.provides("sig")
    # def audio_pipeline(wav):
    #     sig = sb.dataio.dataio.read_audio(wav)
    #     info = torchaudio.info(wav)
    #     resampled = torchaudio.transforms.Resample(
    #         info.sample_rate, hparams["sample_rate"],
    #     )(sig)
    #     return resampled

    # sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 2. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "tokens"],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, = dataio_prepare(
        hparams
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    exit()
    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )