#!/usr/bin/env/python3
"""
ASR helper functions for the SLU recipe.

Authors
 * Loren Lugosch 2020
"""

import os
import torch
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece


class ASR(sb.Brain):
    def transcribe(self, wavs, wav_lens):
        with torch.no_grad():
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            feats = self.hparams.compute_features(wavs)
            feats = self.hparams.normalize(feats, wav_lens)
            encoder_out = self.hparams.enc(feats)
            predicted_tokens, scores = self.hparams.beam_searcher(
                encoder_out, wav_lens
            )

            # Check for and fix hypotheses of length 0, which will cause a division-by-zero error in the loss function.
            for t in predicted_tokens:
                if len(t) == 0:
                    t += [0]

            predicted_tokens_lens = torch.tensor(
                [len(t) for t in predicted_tokens]
            ).float()
            predicted_tokens_lens /= predicted_tokens_lens.max()
            predicted_words = self.hparams.tokenizer(
                predicted_tokens, task="decode_from_list"
            )
            # print(predicted_words[0])
            # print("transcript lengths (in tokens): ", [len(t) for t in predicted_tokens])

            # Pad examples to have same length.
            max_length = max([len(t) for t in predicted_tokens])
            for t in predicted_tokens:
                t += [0] * (max_length - len(t))
            predicted_tokens = torch.tensor([t for t in predicted_tokens])
        return predicted_tokens, predicted_tokens_lens, predicted_words

    def load_tokenizer(self):
        """Loads the sentence piece tokinizer specified in the yaml file"""
        save_model_path = self.hparams.save_folder + "/tok_unigram.model"

        if hasattr(self.hparams, "tok_mdl_file"):
            self.hparams.tokenizer.sp.load(save_model_path)

    def load_lm(self):
        """Loads the LM specified in the yaml file"""
        save_model_path = os.path.join(
            self.hparams.output_folder, "save", "lm_model.ckpt"
        )
        state_dict = torch.load(save_model_path)
        self.hparams.lm_model.load_state_dict(state_dict, strict=True)
        self.hparams.lm_model.eval()


def get_asr_brain():
    print("Loading ASR model...")
    hparams_file = "asr.yaml"
    overrides = {}
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)

    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        model_type=hparams["token_type"],
        character_coverage=1.0,
    )
    hparams["tokenizer"] = tokenizer

    # Brain class initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.load_tokenizer()
    if hasattr(asr_brain.hparams, "lm_ckpt_file"):
        asr_brain.load_lm()
    asr_brain.checkpointer.recover_if_possible()

    return asr_brain
