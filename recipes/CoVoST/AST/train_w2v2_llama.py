#!/usr/bin/env python3
"""Recipe for training a wavlm-large plus LLaMA speech translation system on CoVoST.
The system employs a wavlm-large encoder and a LLaMA decoder.
A simple projection concatenating frames is trained between wavlm-large and LLaMA.

Author
------
 * Titouan Parcollet 2025
"""
import sys

import torch
import torchaudio
import transformers
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class AST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_prompt_translation, tokens_prompt_translation_len = (
            batch.tokens_prompt_translation
        )  # Includes prompt and translation
        prompt_len = batch.prompt_len

        # Turn padding in the speech to zero. We need to do this in case of leaks,
        # because LLAMA padding is using int of value 120k+ which will corrupt A LOT the signal in case of leak.
        audio_len = wavs.shape[1]
        abs_len = torch.round(wav_lens * audio_len)
        audio_attn_mask = length_to_mask(abs_len)
        wavs = wavs * audio_attn_mask

        # Add waveform augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "wav_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        wavs = self.hparams.normalize(wavs, wav_lens)

        # Forward Speech Modules
        feats = self.modules.wav2vec2(wavs, wav_lens)
        down_feats = self.modules.feat_downsampler(feats)
        down_feats_proj = self.modules.proj(down_feats)

        # Format input for LLM; [ audio emb ] + [ prompt emb ] + [ translation emb ]
        # First get relevant lengths
        audio_len = down_feats_proj.shape[1]
        text_len = tokens_prompt_translation.shape[1]
        audio_prompt_len = (audio_len + prompt_len)[0]

        # Then for the full embedding prompt sequence
        if hasattr(self.modules.llm, "module"):
            embeddings = self.modules.llm.module.model.get_input_embeddings()
        else:
            embeddings = self.modules.llm.model.get_input_embeddings()

        inputs_embeds = torch.cat(
            (down_feats_proj, embeddings(tokens_prompt_translation)), dim=1
        )

        # Prepare attn_mask for audio and text and combine them.
        # This is not streaming compatible.
        # For HF to work, masked frames should be 0.
        text_abs_len = torch.round(tokens_prompt_translation_len * text_len)
        abs_len = torch.round(wav_lens * audio_len)
        audio_attn_mask = length_to_mask(abs_len)
        text_attn_mask = length_to_mask(text_abs_len)
        attn_mask = torch.cat([audio_attn_mask, text_attn_mask], dim=-1)

        # LLM forward
        llm_logits = self.modules.llm(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask
        ).logits

        # output layer for seq2seq log-probabilities
        p_seq = self.hparams.log_softmax(llm_logits)

        if hasattr(self.modules.llm, "module"):
            gen_func = self.modules.llm.module.model.generate
        else:
            gen_func = self.modules.llm.model.generate

        # Running decoding if not training
        if stage == sb.Stage.TRAIN:
            hyps = None

        elif stage == sb.Stage.VALID:
            hyps = gen_func(
                inputs_embeds=inputs_embeds[
                    :, :audio_prompt_len
                ],  # give model audio features and prompt for inference
                attention_mask=attn_mask[:, :audio_prompt_len],
                generation_config=self.val_decoding_config,
            )
        elif stage == sb.Stage.TEST:
            hyps = gen_func(
                inputs_embeds=inputs_embeds[
                    :, :audio_prompt_len
                ],  # give model audio features and prompt for inference
                attention_mask=attn_mask[:, :audio_prompt_len],
                generation_config=self.test_decoding_config,
            )

        return p_seq, wav_lens, hyps, audio_prompt_len

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (
            p_seq,
            wav_lens,
            predicted_tokens,
            audio_prompt_len,
        ) = predictions

        ids = batch.id
        tokens_translation, tokens_translation_len = batch.tokens_translation

        # Translation loss
        # We are only interested in computing the loss over the logits after
        # the audio + prompt embeddings. Tokens_translation does not start with bos,
        # so we just need to make sure to shift the logits to the last token of the prompt
        # (to ensure next word prediction)
        p_seq_translation_only = p_seq[:, audio_prompt_len - 1 :]

        loss = self.hparams.nll_loss(
            p_seq_translation_only,
            tokens_translation,
            length=tokens_translation_len,
        )

        if stage != sb.Stage.TRAIN:

            # Removing the eos
            predictions = self.tokenizer.batch_decode(predicted_tokens)
            targets = self.tokenizer.batch_decode(tokens_translation)
            predictions = remove_after_eos(predictions)
            targets = remove_after_eos(targets)

            self.bleu_metric.append(ids, predictions, [targets])

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(
                p_seq_translation_only,
                tokens_translation,
                tokens_translation_len,
            )

        return loss

    def init_optimizers(self):
        self.optimizer = self.hparams.Adam(self.hparams.model.parameters())

        self.optimizers_dict = {"model_optimizer": self.optimizer}

        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.Adam_wav2vec2(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

        # Define generation config depending on runtime values
        self.val_decoding_config = transformers.GenerationConfig(
            num_beams=self.hparams.valid_beam_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=500,
        )

        # Define generation config depending on runtime values
        self.test_decoding_config = transformers.GenerationConfig(
            num_beams=self.hparams.test_beam_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=500,
        )

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
            stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()
            stage_stats["ACC"] = self.acc_metric.summarize()

            if (
                self.optimizer_step > self.hparams.warmup_steps
                and not self.hparams.wav2vec2_frozen
            ):
                (
                    old_lr_wav2vec,
                    new_lr_wav2vec,
                ) = self.hparams.lr_annealing_wav2vec(stage_stats["ACC"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
            else:
                old_lr_wav2vec = (
                    self.hparams.lr_annealing_wav2vec.hyperparam_value
                )

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "lr_wav2vec": old_lr_wav2vec,
                "steps": steps,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=3,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def remove_after_eos(list_of_str, eos_wrd="<|end_of_text|>"):
    """Remove all the text after EOS to obtain the clean translation. Receives a list of string e.g. ['the cat<|end_of_text|>[PAD]']"""
    cleaned = []
    for line in list_of_str:
        index = line.find(eos_wrd)
        if index != -1:
            cleaned.append(line[:index])
        else:
            cleaned.append(line)
    return cleaned


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    train_data = train_data.filtered_sorted(
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_val_test"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than_val_test"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:

    # Get the prompt from yaml and tokenize it
    prompt = hparams["llm_prompt"]
    logger.info(f"Using the following prompt: {repr(prompt)}")

    # Don't add EOS after prompt, only add EOS after transcripts
    # Always manually add eos and bos because HF is not consistent.
    eos_token_id = torch.LongTensor([tokenizer.eos_token_id])
    bos_token_id = torch.LongTensor([tokenizer.bos_token_id])

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze()

    prompt_ids = torch.cat([prompt_ids, bos_token_id])

    # We want BOS + prompt + translation + EOS
    @sb.utils.data_pipeline.takes("translation")
    @sb.utils.data_pipeline.provides(
        "translation",
        "tokens_translation",
        "tokens_prompt_translation",
        "prompt_len",
    )
    def st_text_pipeline(translation):
        yield translation
        tokens_translation = tokenizer(
            translation, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze()
        no_eos_trans = tokens_translation
        tokens_translation = torch.cat([tokens_translation, eos_token_id])
        yield tokens_translation
        tokens_prompt_translation = torch.cat((prompt_ids, no_eos_trans))
        yield tokens_prompt_translation
        prompt_len = prompt_ids.size(0)
        yield prompt_len

    sb.dataio.dataset.add_dynamic_item(datasets, st_text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "translation",
            "tokens_translation",
            "tokens_prompt_translation",
            "prompt_len",
        ],
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
        test_data,
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

    # Dataset preparation (parsing CommonVoice)
    from covost_prepare import prepare_covost  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_covost,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "src_language": hparams["src_language"],
            "tgt_language": hparams["tgt_language"],
            "skip_prep": hparams["skip_prep"],
            "convert_to_wav": hparams["convert_to_wav"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["modules"]["llm"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    ast_brain = AST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    ast_brain.tokenizer = tokenizer

    # Manage dynamic batching
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    test_dataloader_opts = hparams["test_dataloader_opts"]
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
    ast_brain.fit(
        ast_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    ast_brain.evaluate(
        valid_data,
        max_key="ACC",
        test_loader_kwargs=test_dataloader_opts,
    )

    ast_brain.evaluate(
        test_data,
        max_key="ACC",
        test_loader_kwargs=test_dataloader_opts,
    )
