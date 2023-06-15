"""Whisper model that supports decoding with predicted batch of languages,
implements a more efficient decoding and whose tokenizer's vocabulary
can be progressively extended by adding new tokens.

Authors
 * Luca Della Libera 2023
"""

import torch
from torch.nn import functional as F
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

    # Wrap in torch.no_grad for more efficient inference
    def generate(
        self,
        wav=None,
        audio_features=None,
        forced_decoder_locale=None,
        max_gen_tokens=445,
        beam_size=1,
        length_norm_coeff=0.0,
        eos_threshold=float("inf"),
        return_all=False,
    ):
        """Generate a transcription via greedy or beam search.

        Parameters
        ---------
        wav : torch.Tensor
            A batch of audio signals to transform to features.
            Either `wav` or `audio_features` argument should be given.
        audio_features : torch.Tensor
            A batch of features.
            Either `wav` or `audio_features` argument should be given.
        forced_decoder_locale: str
            The locale (e.g. "en", "de", "it") to use for decoding
            (the same for all batch elements).
            If not specified, Whisper's predicted locale (one for
            each batch element) is used.
        max_gen_tokens: int
            The maximum number of tokens to generate.
            Low values of `max_gen_tokens` might result in truncated decoded
            sequences but reduce the amount of memory required for decoding
            (useful when `OutOfMemoryError` is raised).
        beam_size: int
            The beam size. Greedy search is used if `beam_size` is 1,
            beam search otherwise.
        length_norm_coeff: float
            The length normalization coefficient for beam search decoding
            (used only if `beam_size` > 1).
        eos_threshold: float
            The EOS threshold for beam search decoding (used only if `beam_size` > 1).
        return_all: bool
            True to return all the hypotheses (`beam_size` for each batch element),
            False to return only the one with the highest score.

        Returns
        -------
        torch.Tensor
            The batch of hypotheses, shape: ['batch_size`, `beam_size`, `seq_length`]
            if `return_all` is True, ['batch_size`, `seq_length`] otherwise.
        torch.Tensor
            The batch of scores, shape: ['batch_size`, `beam_size`]
            if `return_all` is True, ['batch_size`] otherwise.

        Raises
        ------
        ValueError:
            If an invalid argument value is given.

        Examples
        --------
        >>> model_hub = "openai/whisper-tiny"
        >>> save_path = "savedir"
        >>> model = ProgressiveWhisper(model_hub, save_path, sampling_rate=16000)
        >>> inputs = torch.randn([2, 93680])
        >>> with torch.no_grad():
        ...     hyps, scores = model.generate(inputs, beam_size=5)

        """
        if wav is None and audio_features is None:
            raise ValueError(
                "Either `wav` or `audio_features` argument should be given"
            )
        max_target_positions = self.model.config.max_target_positions - 4 + 1
        if max_gen_tokens > max_target_positions:
            raise ValueError(
                f"`max_gen_tokens` ({max_gen_tokens}) must be less than {max_target_positions}"
            )
        if audio_features is None:
            audio_features = self.forward_encoder(wav)
        batch_size = audio_features.shape[0]
        startoftranscript_id = 50258
        transcribe_id = 50359
        notimestamps_id = 50363
        pad_id = self.model.config.pad_token_id
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
            # Handle different Chinese variants
            if forced_decoder_locale.lower().startswith("zh-"):
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

        # Prepare token suppression mask
        suppress_mask = torch.ones(
            self.model.decoder.embed_tokens.num_embeddings,
            device=audio_features.device,
            dtype=torch.bool,
        )
        suppress_mask[self.model.config.suppress_tokens] = False

        if beam_size > 1:
            hyps, scores = self._beam_search(
                audio_features,
                hyps,
                suppress_mask,
                max_gen_tokens,
                beam_size,
                length_norm_coeff,
                eos_threshold,
            )
            if not return_all:
                hyps, scores = hyps[:, 0, :], scores[:, 0]
        else:
            hyps, scores = self._greedy_search(
                audio_features, hyps, suppress_mask, max_gen_tokens,
            )
            if return_all:
                hyps, scores = hyps[:, None, :], scores[:, None]

        return hyps, scores

    def _greedy_search(
        self,
        audio_features,  # B x S x F
        hyps,  # B x T
        suppress_mask,  # K
        max_gen_tokens,
    ):
        endoftext_id = self.tokenizer.eos_token_id
        batch_size = audio_features.shape[0]  # B
        num_gen_tokens = 0
        # B
        alive_mask = torch.ones(
            batch_size, dtype=torch.bool, device=audio_features.device
        )
        # B
        scores = torch.zeros(batch_size, device=audio_features.device)
        # Autoregressive loop
        # B* x S x F
        alive_audio_features = audio_features
        # B*
        alive_scores = scores.clone()
        while True:
            # B* x T
            alive_hyps = hyps[alive_mask, : num_gen_tokens + 4]
            # B* x T x K
            logits, _ = self.forward_decoder(alive_audio_features, alive_hyps)
            # B* x K
            logits = logits[:, -1, :]
            logits[:, ~suppress_mask] = -float("inf")
            log_probs = logits.log_softmax(dim=-1)
            # B*
            log_probs, gen_token_ids = log_probs.max(dim=-1)
            alive_scores += log_probs
            # B*
            hyps[alive_mask, num_gen_tokens + 4] = gen_token_ids
            scores[alive_mask] = alive_scores
            num_gen_tokens += 1
            if num_gen_tokens >= max_gen_tokens:
                break
            # B*
            alive_mask_unchanged = gen_token_ids != endoftext_id
            if not alive_mask_unchanged.all():
                alive_mask[
                    alive_mask == True
                ] = alive_mask_unchanged  # noqa: E712
                if not alive_mask.any():
                    break
                # B* x S x F
                alive_audio_features = audio_features[alive_mask]
                # B*
                alive_scores = scores[alive_mask]
        # B x T
        hyps = hyps[:, 4 : num_gen_tokens + 4]
        return hyps, scores

    def _beam_search(
        self,
        audio_features,  # B x S x F
        hyps,  # B x T
        suppress_mask,  # K
        max_gen_tokens,
        beam_size,  # N
        length_norm_coeff,
        eos_threshold,
    ):
        endoftext_id = self.tokenizer.eos_token_id
        batch_size = audio_features.shape[0]  # B
        num_gen_tokens = 0
        # B
        alive_mask = torch.ones(
            batch_size, dtype=torch.bool, device=audio_features.device
        )
        # B
        scores = torch.zeros(batch_size, device=audio_features.device)
        # N x B x T
        final_hyps = hyps.expand(beam_size, -1, -1).clone()
        # N x B
        final_scores = torch.zeros(
            beam_size, batch_size, device=audio_features.device
        )
        # B
        final_hyps_count = torch.zeros(
            batch_size, dtype=torch.long, device=audio_features.device
        )
        # Autoregressive loop
        while True:
            if num_gen_tokens == 0:
                # B* x S x F
                alive_audio_features = audio_features
                # B* x T
                alive_hyps = hyps[:, : num_gen_tokens + 4]
                alive_batch_size = alive_hyps.shape[0]
                # B* x T x K
                logits, _ = self.forward_decoder(
                    alive_audio_features, alive_hyps
                )
            else:
                # N x B* x T
                alive_hyps = hyps[:, alive_mask, : num_gen_tokens + 4]
                # NB* x T
                alive_hyps = alive_hyps.movedim(0, 1).reshape(
                    beam_size * alive_batch_size, -1
                )
                # NB* x T x K
                logits, _ = self.forward_decoder(
                    alive_audio_features, alive_hyps
                )
            # NB* x K or B* x K (num_gen_tokens=0)
            logits = logits[:, -1, :]
            logits[:, ~suppress_mask] = -float("inf")
            log_probs = logits.log_softmax(dim=-1)
            if eos_threshold < float("inf"):
                # NB* or B* (num_gen_tokens=0)
                max_log_probs, _ = log_probs.max(dim=-1)
                eos_log_probs = log_probs[:, endoftext_id]
                eos_mask = eos_log_probs <= (eos_threshold * max_log_probs)
                log_probs[:, endoftext_id][eos_mask] = -1e20
            if num_gen_tokens == 0:
                # B*
                alive_scores = scores
                # B* x K
                alive_scores = alive_scores[:, None] + log_probs
                # K x B*
                alive_scores = alive_scores.movedim(0, 1)
                # N x B*
                alive_scores, gen_token_ids = alive_scores.topk(
                    beam_size, dim=0
                )
                # N x B x S x F
                audio_features = audio_features.expand(beam_size, -1, -1, -1)
                # N x B x T
                hyps = hyps.expand(beam_size, -1, -1).clone()
                # N x B
                scores = scores.expand(beam_size, -1).clone()

                alive_batch_size = hyps.shape[1]
                # NB* x S x F
                alive_audio_features = audio_features.movedim(0, 1).reshape(
                    beam_size * alive_batch_size,
                    -1,
                    alive_audio_features.shape[-1],
                )
            else:
                # N x B* x K
                log_probs = log_probs.reshape(
                    alive_batch_size, beam_size, -1
                ).movedim(0, 1)
                # N x B* x K
                alive_scores = alive_scores[:, :, None] + log_probs
                if length_norm_coeff > 0.0:
                    alive_scores /= (num_gen_tokens + 1) ** length_norm_coeff
                # N x K x B*
                alive_scores = alive_scores.movedim(-1, 1)
                # NK x B*
                alive_scores = alive_scores.reshape(-1, alive_batch_size)
                # N x B*
                alive_scores, gen_token_ids = alive_scores.topk(
                    beam_size, dim=0
                )
                hyp_idxes = gen_token_ids // logits.shape[-1]
                gen_token_ids -= logits.shape[-1] * hyp_idxes
                # N x B*
                hyp_idxes += torch.arange(
                    0,
                    alive_batch_size * beam_size,
                    beam_size,
                    device=hyp_idxes.device,
                )[None]
                # N x B* x T
                hyps[:, alive_mask, : num_gen_tokens + 4] = alive_hyps[
                    hyp_idxes
                ]
                if length_norm_coeff > 0.0:
                    alive_scores *= (num_gen_tokens + 1) ** length_norm_coeff
            # N x B*
            hyps[:, alive_mask, num_gen_tokens + 4] = gen_token_ids
            # N x B*
            scores[:, alive_mask] = alive_scores
            # N x B*
            endoftext = gen_token_ids == endoftext_id
            num_gen_tokens += 1
            if endoftext.any() or num_gen_tokens == max_gen_tokens:
                alive_final_hyps = final_hyps[:, alive_mask]
                alive_final_scores = final_scores[:, alive_mask]
                start_idxes = final_hyps_count[alive_mask]
                alive_new_hyps_count = endoftext.sum(dim=0)
                end_idxes = (start_idxes + alive_new_hyps_count).clamp(
                    max=beam_size
                )

                idxes_mask = F.one_hot(start_idxes, num_classes=beam_size + 1)
                idxes_mask -= F.one_hot(end_idxes, num_classes=beam_size + 1)
                idxes_mask = idxes_mask.cumsum(dim=-1)[:, :-1].bool().T
                diff_mask = F.one_hot(
                    torch.zeros_like(alive_new_hyps_count),
                    num_classes=beam_size + 1,
                )
                start_mask = diff_mask - F.one_hot(
                    alive_new_hyps_count, num_classes=beam_size + 1
                )
                start_mask = start_mask.cumsum(dim=-1)[:, :-1].bool().T
                diff_mask -= F.one_hot(
                    alive_new_hyps_count.min(beam_size - start_idxes),
                    num_classes=beam_size + 1,
                )
                diff_mask = diff_mask.cumsum(dim=-1)[:, :-1].bool().T

                alive_final_hyps.movedim(0, 1)[idxes_mask.T] = hyps[
                    :, alive_mask
                ].movedim(0, 1)[endoftext.T][diff_mask.T[start_mask.T]]
                alive_final_scores.movedim(0, 1)[idxes_mask.T] = scores[
                    :, alive_mask
                ].movedim(0, 1)[endoftext.T][diff_mask.T[start_mask.T]]

                final_hyps[:, alive_mask] = alive_final_hyps
                final_scores[:, alive_mask] = alive_final_scores
                final_hyps_count[alive_mask] = end_idxes

                alive_scores[endoftext] = -float("inf")
                scores[:, alive_mask] = alive_scores
                if num_gen_tokens >= max_gen_tokens:
                    break

                # B*
                alive_mask_unchanged = end_idxes < beam_size
                if not alive_mask_unchanged.all():
                    alive_mask[
                        alive_mask == True
                    ] = alive_mask_unchanged  # noqa: E712
                    if not alive_mask.any():
                        break
                    # N x B* x S x F
                    alive_audio_features = audio_features[:, alive_mask]
                    # N x B*
                    alive_scores = scores[:, alive_mask]
                    alive_batch_size = alive_scores.shape[1]
                    # NB* x S x F
                    alive_audio_features = alive_audio_features.movedim(
                        0, 1
                    ).reshape(
                        beam_size * alive_batch_size,
                        -1,
                        alive_audio_features.shape[-1],
                    )
        # N x B x T
        final_hyps = final_hyps[:, :, 4 : num_gen_tokens + 4]
        # B x N x T
        final_hyps = final_hyps.movedim(0, 1)
        # B x N
        final_scores = final_scores.movedim(0, 1)
        final_scores, final_score_idxes = final_scores.sort(
            dim=-1, descending=True, stable=True
        )
        final_score_idxes += torch.arange(
            0,
            batch_size * beam_size,
            beam_size,
            device=final_score_idxes.device,
        )[:, None]
        final_hyps = (
            final_hyps.reshape(batch_size * beam_size, -1)[
                final_score_idxes.movedim(0, 1)
            ]
            .reshape(beam_size, batch_size, -1)
            .movedim(0, 1)
        )
        return final_hyps, final_scores
