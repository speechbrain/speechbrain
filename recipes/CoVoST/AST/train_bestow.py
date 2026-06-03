#!/usr/bin/env python3
"""Recipe for training a Transformer AST system with CoVoST as described by
BEST-OW (https://arxiv.org/abs/2406.19954).

The system employs a streaming encoder (BESTRQ), a decoder (LlaMA 3) and a
self-attention+cross attention layer as a projection layer, named mixing decoder.

Authors
 * Titouan Parcollet 2025
"""

import os
import string
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.TransformerASR import (
    make_transformer_src_tgt_masks,
)
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class AST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        prompt_ids, prompt_ids_lens = batch.prompt_ids
        tokens_prompt_translation, tokens_prompt_translation_len = (
            batch.tokens_prompt_translation
        )  # Includes prompt and transcript
        prompt_len = batch.prompt_len[0]

        if self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
            # We add this to avoid changing the whole scorer by adding an extra
            # arg. This is used for delay steps with prompt decoding, so not
            # really about DCT but more about decoding..
            if dynchunktrain_config is not None:
                dynchunktrain_config.delay_steps = prompt_len
        else:
            dynchunktrain_config = None

        # Turn padding in the speech to zero. We need to do this in case of leaks,
        # because LLAMA padding is using int of value 120k+ which may corrupt the signal.
        audio_len = wavs.shape[1]
        abs_len = torch.round(wav_lens * audio_len)
        audio_attn_mask = length_to_mask(abs_len)
        wavs = wavs * audio_attn_mask

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            if self.optimizer_step > self.hparams.augment_warmup:
                feats, _ = self.hparams.fea_augment(feats, wav_lens)

        # Forward Speech Modules
        feats = self.modules.CNN(feats)
        enc_out = self.modules.enc(
            feats, wav_lens, dynchunktrain_config=dynchunktrain_config
        )

        # Downsample the features (and dynchunk config) in Time.
        enc_out = self.modules.feat_downsampler(enc_out)

        if dynchunktrain_config is not None:
            dynchunktrain_config = DynChunkTrainConfig(
                chunk_size=dynchunktrain_config.chunk_size
                // self.hparams.downsampling_factor,
                left_context_size=dynchunktrain_config.left_context_size,
                warmup_chunks=dynchunktrain_config.warmup_chunks,
            )
            dynchunktrain_config.delay_steps = prompt_len

        # DDP compliant.. We must project the embedding dim to the mixing dim.
        # This is so ugly, hopefully we can find a way to fix it some days.
        if hasattr(self.modules.llm, "module"):
            text_embeds = self.modules.llm.module.embed_tokens(
                tokens_prompt_translation
            )
        else:
            text_embeds = self.modules.llm.embed_tokens(
                tokens_prompt_translation
            )

        text_embeds_proj = self.modules.text_embeds_proj(text_embeds)

        # The mixing_decoder is a TransformerDecoder.
        # We prepare the masks for the cross-attention before the LLM
        (
            enc_key_padding_mask,
            dec_key_padding_mask,
            enc_mask,
            dec_mask,
        ) = make_transformer_src_tgt_masks(
            enc_out,
            tokens_prompt_translation,
            wav_lens,
            causal=self.hparams.mixing_causal,  # dec self-att is causal
            pad_idx=self.hparams.pad_token,
            dynchunktrain_config=dynchunktrain_config,  # dec cross-attn.
            delay_steps=prompt_len,
        )

        # This is due to streaming. When we are using streaming training, a new mask
        # must exist between the cross attention of the mixing decoder and the
        # output of the speech encoder. If not, then there is no mask. Hence
        # why memory_mask is None in that case (memory is the speech encoder output).
        if isinstance(dec_mask, tuple):
            tgt_mask, memory_mask = dec_mask
        else:
            tgt_mask = dec_mask
            memory_mask = None

        mixed_output = self.modules.mixing_decoder(
            text_embeds_proj,
            enc_out,
            tgt_key_padding_mask=dec_key_padding_mask,  # Masked frames should be 1
            memory_key_padding_mask=enc_key_padding_mask,  # Masked frames should be 1
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        # Proj back to LLM embedding dim and residual connection
        mixed_embs = self.modules.llm_proj(mixed_output[0]) + text_embeds

        # LLM forward
        llm_logits = self.modules.llm(
            inputs_embeds=mixed_embs, attention_mask=~dec_key_padding_mask
        ).logits  # Masked frames should be 0

        # output layer for seq2seq log-probabilities
        p_seq = self.hparams.log_softmax(llm_logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.greedy_searcher(
                enc_out.detach(),
                wav_lens,
                memory=prompt_ids,
                keep_lens_rel=True,
                dynchunk_config=dynchunktrain_config,
            )

        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.beam_searcher(
                enc_out.detach(),
                wav_lens,
                memory=prompt_ids,
                dynchunk_config=dynchunktrain_config,
            )

        return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (
            p_seq,
            wav_lens,
            predicted_tokens,
        ) = predictions

        ids = batch.id
        tokens_translation, tokens_translation_len = batch.tokens_translation
        prompt_len = batch.prompt_len[0]

        # Translation loss
        # We are only interested in computing the loss over the logits after
        # the prompt embeddings. Tokens_translation does not start with bos,
        # so we just need to make sure to shift the logits to the last token
        # of the prompt (to ensure next word prediction)

        p_seq_translation_only = p_seq[:, prompt_len - 1 :]

        loss = self.hparams.nll_loss(
            p_seq_translation_only,
            tokens_translation,
            length=tokens_translation_len,
        )

        if stage != sb.Stage.TRAIN:
            # Removing the eos
            predictions = self.tokenizer.batch_decode(predicted_tokens)
            targets = self.tokenizer.batch_decode(tokens_translation)
            predictions = remove_after_eos(
                predictions, eos_wrd=self.hparams.eos_word
            )
            targets = remove_after_eos(targets, eos_wrd=self.hparams.eos_word)

            # Lazy logging
            if if_main_process() and stage == sb.Stage.TEST:
                source = batch.transcription
                write_to_file(
                    source, predictions, targets, self.hparams.save_folder
                )

            targets = remove_punctuation(targets)
            predictions = remove_punctuation(predictions)

            self.bleu_metric.append(ids, predictions, [targets])

            # compute the accuracy
            self.acc_metric.append(
                p_seq_translation_only,
                tokens_translation,
                tokens_translation_len,
            )

        return loss

    def init_optimizers(self):
        self.optimizer = self.hparams.Adam(self.hparams.model.parameters())
        self.optimizers_dict = {"model_optimizer": self.optimizer}

        self.bestrq_lora_optimizer = self.hparams.Adam_ft(
            self.hparams.bestrq_lora.parameters()
        )
        self.optimizers_dict["bestrq_lora_optimizer"] = (
            self.bestrq_lora_optimizer
        )
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "bestrq_lora_opt", self.bestrq_lora_optimizer
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.cosine_annealing(self.optimizer)
            self.hparams.cosine_ft_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
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

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            lr = self.hparams.cosine_annealing.current_lr
            lr_lora = self.hparams.cosine_ft_annealing.current_lr
            steps = self.optimizer_step

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "lr_bestrq_lora": lr_lora,
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
                num_to_keep=1,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def remove_punctuation(list_of_str):
    """Remove all punctuation marks."""
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans("", "", string.punctuation)

    # Remove punctuation from each string in the list
    return [s.translate(translator) for s in list_of_str]


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


def write_to_file(source, list_str_pred, list_str_target, save_folder):
    """Small utility function to be remove that dumps to file predictions and target translations.
    To run under DDP protection scope!

    """
    filename = os.path.join(save_folder, "translation_out.txt")

    with open(filename, "a", encoding="utf-8") as file:
        for i in range(len(list_str_pred)):
            file.write(source[i] + "\n")
            file.write(list_str_target[i] + "\n")
            file.write(list_str_pred[i] + "\n")
            file.write("--------------------------\n")


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
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:

    # Get the prompt from yaml and tokenize it
    prompt = hparams["llm_prompt"]
    logger.info(f"Using the following prompt: {repr(prompt)}")

    # Don't add EOS after prompt, only add EOS after translation
    # Always manually add eos and bos because HF is not consistent.
    eos_token_id = torch.LongTensor([tokenizer.eos_token_id])
    bos_token_id = torch.LongTensor([tokenizer.bos_token_id])

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze()

    prompt_bos = torch.cat([prompt_ids, bos_token_id])

    # We want BOS + prompt + translation + EOS
    @sb.utils.data_pipeline.takes("translation", "transcription")
    @sb.utils.data_pipeline.provides(
        "transcription",
        "translation",
        "tokens_translation",
        "tokens_prompt_translation",
        "prompt_ids",
        "prompt_len",
    )
    def st_text_pipeline(translation, transcription):
        yield transcription
        yield translation
        tokens_translation = tokenizer(
            translation, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze()
        no_eos_trans = tokens_translation
        tokens_translation = torch.cat([tokens_translation, eos_token_id])
        yield tokens_translation
        tokens_prompt_translation = torch.cat((prompt_bos, no_eos_trans))
        yield tokens_prompt_translation
        prompt_len = prompt_bos.size(0)
        yield prompt_ids
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
            "prompt_ids",
            "prompt_len",
            "transcription",
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

    # Load the pretrained model
    if (
        "pretrainer" in hparams.keys()
        and hparams["bestrq_model_path"] is not None
    ):
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

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
