"""Whisper model that supports decoding with predicted batch of languages,
implements a more efficient greedy decoding and whose tokenizer's vocabulary
can be progressively extended by adding new tokens.

Authors
 * Luca Della Libera 2022
"""

import torch
from transformers.models.whisper.tokenization_whisper import (
    LANGUAGES,
    TASK_IDS,
    TO_LANGUAGE_CODE,
    WhisperTokenizer,
)

from speechbrain.lobes.models.huggingface_whisper import HuggingFaceWhisper


__all__ = [
    "ProgressiveWhisper",
]


class ProgressiveWhisperTokenizer(WhisperTokenizer):
    # override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_languages = {k: v for k, v in LANGUAGES.items()}  # Copy
        self.to_language_codes = {
            k: v for k, v in TO_LANGUAGE_CODE.items()
        }  # Copy

    # override
    @property
    def prefix_tokens(self):
        # all_special_ids = self.all_special_ids
        bos_token_id = 50258  # all_special_ids[-106]
        translate_token_id = 50358  # all_special_ids[-6]
        transcribe_token_id = 50359  # all_special_ids[-5]
        notimestamps_token_id = 50363  # all_special_ids[-1]
        # langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in self.to_language_codes:
                language_id = self.to_language_codes[self.language]
            else:
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be in: {self.to_language_codes.keys()}"
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(
                    f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}"
                )

        bos_sequence = [bos_token_id]
        if self.language is not None:
            # Need to replace with custom code because language ID is hardcoded...
            # bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
            bos_sequence.append(
                self.encode(f"<|{language_id}|>", add_special_tokens=False)[0]
            )
        if self.task is not None:
            bos_sequence.append(
                transcribe_token_id
                if self.task == "transcribe"
                else translate_token_id
            )
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence


class ProgressiveWhisper(HuggingFaceWhisper):
    # override
    def __init__(
        self, source, save_path, **kwargs,
    ):
        super().__init__(
            source, save_path, **kwargs,
        )
        if self.tokenizer is not None:
            self.tokenizer = ProgressiveWhisperTokenizer.from_pretrained(
                source,
                language=None,
                task="transcribe",
                predict_timestamps=False,
            )
            # The number of embeddings is 51865 while the vocabulary size is 50364
            # The missing tokens are timestamp tokens (see https://github.com/openai/whisper/discussions/361)
            # To avoid problems when extending the tokenizer and/or the model we add them explicitly
            vocab_size = len(self.tokenizer.get_vocab())
            num_embeddings = self.model.decoder.embed_tokens.num_embeddings
            num_missing_tokens = num_embeddings - vocab_size
            timestamps = [
                i * 30.0 / (num_missing_tokens - 1)
                for i in range(num_missing_tokens)
            ]
            timestamp_tokens = [f"<|{ts:.2f}|>" for ts in timestamps]
            self.tokenizer.add_tokens(timestamp_tokens)

    @torch.no_grad()
    def generate(
        self,
        wav=None,
        audio_features=None,
        forced_decoder_locale=None,
        max_gen_tokens=445,
        strategy="greedy",
    ):
        if wav is None and audio_features is None:
            raise ValueError(
                "Either `wav` or `audio_features` argument should be given"
            )
        if audio_features is None:
            audio_features = self.forward_encoder(wav)
        batch_size = audio_features.shape[0]
        (
            startoftranscript_id,
            transcribe_id,
            notimestamps_id,
        ) = self.tokenizer.prefix_tokens
        pad_id = self.model.config.pad_token_id
        endoftext_id = self.tokenizer.eos_token_id

        hyps = torch.full(
            (batch_size, max_gen_tokens + 4),
            pad_id,
            dtype=torch.long,
            device=audio_features.device,
        )
        if forced_decoder_locale is None:
            # Compute most likely language token IDs
            all_lang_tokens = [
                f"<|{l}|>" for l in self.tokenizer.supported_languages
            ]
            all_lang_tokens_ids = self.tokenizer.convert_tokens_to_ids(
                all_lang_tokens
            )
            hyps[:, 0] = startoftranscript_id
            logits, _ = self.forward_decoder(audio_features, hyps[:, :1])
            lang_mask = torch.zeros(
                logits.shape[-1], device=logits.device, dtype=torch.bool
            )
            lang_mask[all_lang_tokens_ids] = True
            logits[:, :, ~lang_mask] = -float("inf")
            lang_tokens_ids = logits.argmax(dim=-1)[:, 0]
        else:
            if forced_decoder_locale.lower() == "zh-cn":
                forced_decoder_locale = "zh"
            if (
                forced_decoder_locale.lower()
                not in self.tokenizer.supported_languages
            ):
                raise NotImplementedError(
                    f"Unsupported language: {forced_decoder_locale}"
                )
            lang_tokens_ids = self.tokenizer.convert_tokens_to_ids(
                f"<|{forced_decoder_locale.lower()}|>"
            )

        # Prepare initial tokens in the right format
        hyps[:, 0] = startoftranscript_id
        hyps[:, 1] = lang_tokens_ids
        hyps[:, 2] = transcribe_id
        hyps[:, 3] = notimestamps_id

        # Autoregressive loop
        num_gen_tokens = 0
        unfinished_mask = torch.ones(
            len(hyps), dtype=torch.bool, device=audio_features.device
        )
        while True:
            logits, _ = self.forward_decoder(
                audio_features[unfinished_mask],
                hyps[unfinished_mask, : num_gen_tokens + 4],
            )
            # Prepare suppress mask
            suppress_mask = torch.ones(
                logits.shape[-1], device=audio_features.device, dtype=torch.bool
            )
            suppress_mask[self.model.config.suppress_tokens] = False
            logits[:, :, ~suppress_mask] = -float("inf")
            gen_tokens = logits.argmax(dim=-1)[:, -1]
            hyps[unfinished_mask, num_gen_tokens + 4] = gen_tokens
            unfinished_mask[unfinished_mask == True] = (
                gen_tokens != endoftext_id
            )
            num_gen_tokens += 1
            if (not unfinished_mask.any()) or (
                num_gen_tokens >= max_gen_tokens
            ):
                break
        return hyps[:, 4 : num_gen_tokens + 3]
