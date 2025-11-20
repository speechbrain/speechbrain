#!/usr/bin/env python3
"""
module load gcc arrow/17.0.0
module load python/3.12.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install torch==2.7.1
pip install torchaudio==2.7.1
pip install transformers
pip install triton==3.0.0
pip install flash-attn
pip install wandb 
pip install accelerate
pip install -e . 

cd $SLURM_TMPDIR
scp -r $SCRATCH/models/wavlm-large/ .
scp -r $SCRATCH/models/HuggingFaceTB/SmolLM2-360M/ .
scp $HOME/projects/def-ravanelm/datasets/librispeech/*.tar.gz .
for f in *.tar.gz; do tar -xf "$f"; done

cd /home/adelmou/proj/speechbrain/speechllm_librispeech/speechbrain/recipes/LibriSpeech/ASR/transformer/


TOKENIZERS_PARALLELISM=false python train_speechllm.py hparams/llama.yaml         --data_folder $SLURM_TMPDIR/LibriSpeech/         --output_folder $SLURM_TMPDIR/wavlm_large-smollm2_360M_linear_proj_lora         --ssl_hub $SLURM_TMPDIR/wavlm-large/         --llm_path $SLURM_TMPDIR/SmolLM2-360M/         --llm_emb_size 960 --pad_token 128004        --feats_cache_dir $SCRATCH/cached_feats_ls960h --max_batch_length_train 400 --grad_accumulation_factor 8

Authors
 * Adel Moumen 2025
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.integrations.hdf5.cached_item import CachedHDF5DynamicItem

logger = get_logger(__name__)


def get_multimodal_attention_mask(wav, wav_lens, txt, txt_lens, device):
    batch_size = wav.size(0)
    wav_len = wav.size(1)
    txt_len = txt.size(1)
    # Max total length for padding
    max_total_len = wav_len + txt_len
    attention_mask = torch.zeros(batch_size, max_total_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        actual_wav_len = int(wav_lens[i].item() * wav_len)
        actual_txt_len = int(txt_lens[i].item() * txt_len)
        # Fill mask: audio part
        attention_mask[i, :actual_wav_len] = True
        # Fill mask: text part (after audio)
        attention_mask[i, wav_len:wav_len + actual_txt_len] = True
    return attention_mask


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        prompt_len = batch.prompt_len
        
        if self.hparams.use_ssl_feats:
            audio_feats, audio_feats_lens = batch.feats
        else:
            wavs, wav_lens = batch.sig
            wavs = self.hparams.normalize(wavs, wav_lens)
            audio_feats = self.modules.ssl(wavs, wav_lens)
            audio_feats_lens = wav_lens
        # R^L*D -> R^(L/R)*(D*R)
        audio_down_feats = self.modules.feat_downsampler(audio_feats)
        # R^D' -> R^llm_emb_size
        projected_audio_feats = self.modules.proj(audio_down_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal_embds = torch.cat([
            txt_embds[:, 0].unsqueeze(1), # B, D -> B, 1, D
            projected_audio_feats, 
            txt_embds[:, 1:]
        ], dim=1)
        # attention_mask should be all the true audio features + all the true text features
        attention_mask = get_multimodal_attention_mask(
            projected_audio_feats, audio_feats_lens, txt_embds, tokens_bos_lens, self.device
        )
        logits = self.modules.llm(
            inputs_embeds=multimodal_embds, 
            attention_mask=attention_mask
        ).logits
        
        hyps = None        
        if stage != sb.Stage.TRAIN:
            audio_and_prompt_len = projected_audio_feats.shape[1] + prompt_len[0]
            inputs_embeds = multimodal_embds[
                :, :audio_and_prompt_len
            ]
            hyps = self.modules.valid_search(
                inputs_embeds,
                audio_feats_lens,
                attention_mask[:, :audio_and_prompt_len],
            )
        return logits, hyps, projected_audio_feats.shape[1]

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        logits, hyps, _ = predictions
        tokens_eos, _ = batch.tokens_eos
        ids = batch.id

        num_audio_feats = logits.shape[1] - tokens_eos.shape[1]
        # We prepend `-100` to the tokens_eos to ignore them in the loss.
        # This corresponds to the audio features.
        target_tokens = torch.cat([
            torch.full((tokens_eos.shape[0], num_audio_feats), -100, device=self.device),
            tokens_eos,
        ], dim=1).long()
        # compute the cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_tokens.view(-1),
            ignore_index=-100
        )
        if stage != sb.Stage.TRAIN:
            # replace -100 with pad token
            target_tokens = target_tokens.masked_fill(target_tokens == -100, self.tokenizer.pad_token_id)
            preds = self.tokenizer.batch_decode(hyps[0], skip_special_tokens=True)
            preds_words = [pred.split(" ") for pred in preds]
            targets = self.tokenizer.batch_decode(target_tokens, skip_special_tokens=True)
            targets_words = [target.split(" ") for target in targets]
            self.cer_metric.append(ids, preds_words, targets_words)
            self.wer_metric.append(ids, preds_words, targets_words)
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        # check if txt_embedding is already set
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        self.optimizer = self.hparams.opt(self.hparams.model.parameters())
        self.optimizers_dict = {"model_optimizer": self.optimizer}

        if not self.hparams.ssl_frozen:
            self.wav2vec_optimizer = self.hparams.opt_wav2vec2(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_optimizer", self.optimizer)
            self.checkpointer.add_recoverable("wav2vec_optimizer", self.wav2vec_optimizer)

def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
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
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # todo: bos, eos, pad should be set by the user! they are hard to retrieve automatically
    bos_index = tokenizer.bos_token_id
    if bos_index is None:
        bos_index = 128000
    eos_index = tokenizer.eos_token_id
    pad_index = tokenizer.pad_token_id
    start_of_audio_index = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    end_of_audio_index = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    prompt = "Transcribe speech to text."
    print(bos_index, eos_index, pad_index, start_of_audio_index, end_of_audio_index, prompt)
    # exit()
    # print(tokenizer.additional_special_tokens)
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.view(-1).tolist()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "prompt_len"
    )
    def text_pipeline(wrd):
        # wrd = wrd[0] + wrd[1:].lower()
        yield wrd
        tokens_list = tokenizer(wrd, add_special_tokens=False).input_ids
        yield tokens_list
        tokens_bos = torch.LongTensor([start_of_audio_index] + [end_of_audio_index] + prompt_ids  + [bos_index] + tokens_list )
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        prompt_len = len([start_of_audio_index] + [end_of_audio_index] + prompt_ids)
        yield prompt_len

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "prompt_len"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )

def offline_feats_dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]
    
    # todo: bos, eos, pad should be set by the user! they are hard to retrieve automatically
    bos_index = tokenizer.bos_token_id
    if bos_index is None:
        bos_index = 128000
    eos_index = tokenizer.eos_token_id
    pad_index = tokenizer.pad_token_id
    start_of_audio_index = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    end_of_audio_index = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    prompt = "Transcribe speech to text."
    print(bos_index, eos_index, pad_index, start_of_audio_index, end_of_audio_index, prompt)
    # exit()
    # print(tokenizer.additional_special_tokens)
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.view(-1).tolist()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "prompt_len"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer(wrd, add_special_tokens=False).input_ids
        yield tokens_list
        tokens_bos = torch.LongTensor([start_of_audio_index] + [end_of_audio_index] + prompt_ids  + [bos_index] + tokens_list )
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        prompt_len = len([start_of_audio_index] + [end_of_audio_index] + prompt_ids)
        yield prompt_len


    @CachedHDF5DynamicItem.cache(hparams["feats_cache_dir"])
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("feats")
    def feats_pipeline(uid):
        # this function is not used. 
        return uid

    feats_pipeline.change_file_mode("r")
    dynamic_items = [text_pipeline, feats_pipeline]
    output_keys = ["id", "wrd", "tokens_bos", "tokens_eos", "tokens", "prompt_len", "feats"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
        dynamic_items=dynamic_items,
        output_keys=output_keys,
    )

    
    # Build valid dataset with its own cached wrapper
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
        dynamic_items=dynamic_items,
        output_keys=output_keys,
    )


    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, 
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

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
    tokenizer = hparams["llm"].tokenizer
    
    if hparams["use_ssl_feats"]:
        (
            train_data,
            valid_data,
            test_datasets,
            tokenizer,
            train_bsampler,
            valid_bsampler,
        ) = offline_feats_dataio_prepare(hparams, tokenizer)
    else:
        (
            train_data,
            valid_data,
            test_datasets,
            tokenizer,
            train_bsampler,
            valid_bsampler,
        ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # asr_brain.modules.llm = torch.compile(asr_brain.modules.llm)
    asr_brain.tokenizer = tokenizer
    # gen_func = self.modules.llm.module.model.generate
    asr_brain.gen_func = asr_brain.raw_modules.llm.model.generate
    asr_brain.txt_embedding = asr_brain.raw_modules.llm.model.get_input_embeddings()
    # adding objects to trainer:
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
